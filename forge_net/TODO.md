To do list for getting paper published

- [x] Verify unseeded works with MSE
- [x] Run domain decomp (1024) vs unmasked (1024)
- [ ] Determine if RES layers do anything - remove if they do not
- [x] Create normalized vector field plots colored by magnitude (strain)
- [ ] Create vector plots of predicted delta to actual delta (is error in direction random?) (ask Mike)
- [x] Compare chamfer loss versus MSE
- [x] Create timeline plot of hits GT vs pred using MSE Loss
- [ ] Create timeline plot of hits GT vs pred using Chamfer Loss
- [ ] Loss versus press_window - show learning rate (table for next unit of work)
- [ ] Verify any differences between loss_fn(delta_t, delta_tp1) and loss_fn(x_t, x_tp1)
- [ ] Need clearer separation of responsiblity in class design AND imports (don't import *)