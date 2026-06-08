# Changelog

## Unreleased -
- Add `autograd.WarpKernelsFunction` for arbitrary warp kernels (single call)
- Add model to `autograd.UpdateFunction` args
- Change `Environment.render()` to use `state or self.state_0`
- Rename `kinematic_fk` to `eval_kinematic_fk`
- Add edge_count, tri_count, spring_count to `Environment.print_model_info()`
- Add `Environment.data_dir` param, defaults to `rewarped/data/`
- Move `Environment.update()` impl. to `warp_utils.sim_update_inplace()`
- Add `warp_examples.granular`
- Add `warp_examples.quadruped`

## 1.3.3 - 2025-03-13
- Fix MPM asset root path and missing DexDeform asset ([GH-17](https://github.com/rewarped/rewarped/issues/17))

## 1.3.1 - 2025-03-01
- Fix `warp_examples.cloth_throw` regression with `ModelBuilder.add_cloth_grid()`

## 1.3.0 - 2025-02-28
- Initial release
