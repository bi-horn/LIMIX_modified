NumPyLIMIX
Copyright 2024-2026 B.M. Horn, C. Lippert

This project is a derivative work incorporating code from:

1. LIMIX (Apache 2.0)
   Copyright 2018 C. Lippert, D. Horta, F. P. Casale, and O. Stegle
   https://github.com/limix/limix
   Licensed under the Apache License, Version 2.0
   See THIRD_PARTY_LICENSES/LIMIX_APACHE_LICENSE.txt

2. GLIMIX-core (MIT)
   Copyright 2018 Danilo Horta
   https://github.com/limix/glimix-core
   Licensed under the MIT License
   See THIRD_PARTY_LICENSES/GLIMIX_MIT_LICENSE.txt

Modifications include:
- Updated deprecated dependencies
- Python 3.9+ compatibility fixes
- Modified functions in folder "glimix_core_mod" to import a modified function and/or use a refactored initialization
- Added CLI support and phenotype simulator for runtime analysis and comparison with TorchLIMIX
- run_numpylimix included for comparison purposes with the TorchLIMIX pipeline 

Modifications and additions by B.M. Horn, and C. Lippert are licensed under MIT.