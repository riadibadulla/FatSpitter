
# FatSpitter
The script which takes any PyTorch sequential model object and turns it into a FatNet. 

To use call the function fat_spitter() from FatSpitter file.

```Python
import FatSpitter

FatSpitter.fat_spitter(input_channels, output_channels, kernel_size, is_bias=True, pseudo_negativity=False,
                       input_size=28, noise=True)
```
