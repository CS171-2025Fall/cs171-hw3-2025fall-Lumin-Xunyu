#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        //
        //
        // Accumulate radiance
        // assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        // assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        // const Vec3f &L = Li(scene, ray, sampler);
        // camera->getFilm()->commitSample(pixel_sample, L);

        const Vec2f pixel_sample = Vec2f(dx, dy);

        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);

        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      if (scene->hasInfiniteLight())
      {
        const auto& infinite_light = scene->getInfiniteLight();
        SurfaceInteraction dummy;
        color = infinite_light->Le(dummy, ray.direction);
      }
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      //
      // Useful Functions:
      // @see BSDF::sample
      // @see SurfaceInteraction::spawnRay
      //
      // You should update ray = ... with the spawned ray
      Float pdf;
      interaction.bsdf->sample(interaction, sampler, &pdf);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (diffuse_found) {
    color = directLighting(scene, interaction, sampler);
  }

  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  Vec3f color(0, 0, 0);
  const Float light_intensity = 0.2f;
  // 遍历所有点光源
  for (const auto& light : point_lights) {
    Float dist_to_light = Norm(light.position - interaction.p);
    Vec3f light_dir     = Normalize(light.position - interaction.p);
    auto test_ray       = DifferentialRay(interaction.p, light_dir);

    SurfaceInteraction shadow_interaction;
    test_ray.setTimeMax(dist_to_light - EPS);

    // 阴影测试
    if (scene->intersect(test_ray, shadow_interaction)) {
      // 被遮挡，添加环境光
      Vec3f albedo = interaction.bsdf->evaluate(interaction);
      Vec3f ambient = albedo * Vec3f(0.1f, 0.1f, 0.1f) * light_intensity;
      color += ambient;
      continue;  // 跳到下一个光源
    }

    // 未被遮挡，计算直接光照
    const BSDF *bsdf = interaction.bsdf;
    bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

    if (bsdf != nullptr && is_ideal_diffuse) {
      Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);
      Vec3f albedo = bsdf->evaluate(interaction);
      Float attenuation = 1.0f / (dist_to_light * dist_to_light);

      // 累加这个光源的贡献
      color += albedo * light.flux * cos_theta * attenuation * light_intensity;
    }
  }

  // 环境光采样
  if (scene->hasInfiniteLight())
  {
    const auto& env_light = scene->getInfiniteLight();
    const int num_env_samples = 30;
    
    for (int i = 0; i < num_env_samples; ++i)
    {
      // sample() 会修改 interaction.wi，把它设置为指向光源的方向
      SurfaceInteraction env_interaction = env_light->sample(interaction, sampler);
      
      // 正确：使用 interaction.wi（被 sample 修改过的），而不是 env_interaction.wi（默认为0）
      Vec3f wi = Normalize(interaction.wi);
      Float cos_theta = std::max(Dot(wi, interaction.normal), 0.0f);

      // 创建阴影光线测试可见性
      auto shadow_ray = DifferentialRay(interaction.p, wi);
      SurfaceInteraction shadow_interaction;
      
      // 如果光线没有被遮挡，累加环境光贡献
      if (!scene->intersect(shadow_ray, shadow_interaction))
      {
        Vec3f albedo = interaction.bsdf->evaluate(interaction);
        Vec3f Le = env_light->Le(env_interaction, wi);
        // Monte Carlo 估计：除以采样数量
        color += albedo * Le * cos_theta / Float(num_env_samples);
      }
    }
  }

  return color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
