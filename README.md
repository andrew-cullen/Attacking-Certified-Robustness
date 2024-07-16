# Attacking-Certified-Robustness

Repository associated with the ICML (2024) paper "Et Tu Certifications: Robustness Certificates Yield Better Adversarial Examples" by Cullen, A.C. and Liu, S. and Montague, P. and Erfani, S.M. and Rubinstein, B.I.P.

This code demonstrates how adversarial attacks can be constructed against certifiably robust models. It differs from techniques like Expectation over Transformation in that it is designed to attack the input sample and the aggregated output, rather than the individual draws under noise. The attack has also been designed specifically for attacks against certified robustness. 

## Example Run Script (More details will be provided) 

*Training*
```
python3 main.py --train --parallel eval --eval --dataset imagenet --filename imagenet-25 --sigma 0.25 --pgd_radii 10 --new_min_step 0.01 --new_max_step 0.125 > i25_new
```

*Resuming trained code*
```
python3 main.py --ablation --resume 80 --eval --parallel eval --dataset imagenet --filename iout --sigma 1.0 --pgd_radii 200 --new_min_step 0.03921568627451 --new_max_step 1 --samples 750 > output_catch
```

## Dependencies
- PyTorch >=2.0 
- Torchvision
- Numpy
- Statsmodels
- Scipy
- autoattack


**This repository is being successively populated. The code is being uploaded sequentially covering the following features**:
- [x] Certification Aware Attacks
- [x] Attacks against CR for C-W, AutoAttack, and DeepFool.
- [ ] Detailed examples of running code.  
- [ ] Analysis Code.
- [ ] Output data.
- [ ] Results.
- [ ] Detailed Dependencies List.
- [ ] Code for non-randomised smoothing attacks (specifically Interval Bound Propagation).
- [ ] MACER
