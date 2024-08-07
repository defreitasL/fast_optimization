
# IHSetCalibration
Python package to perform the calibration in IH-SET software.

## :house: Local installation
* Using pip:
```bash

pip install git+https://github.com/defreitasL/IHSetCalibration.git

```

---
## :zap: Main methods

* [objective_functions](./IHSetCalibration/objectives_functions.py):
```python
# transform GOW data to standart wac.nc IH-SET file
objective_functions(method, metrics, **kwargs)
```
* [setup_spotpy](./IHSetCalibration/setup_spotpy.py)
```python
# transform GOS data to standart sl.nc IH-SET file
setup_spotpy(model_object)
```



## :package: Package structures
````

IHSetCalibration
|
├── LICENSE
├── README.md
├── build
├── dist
├── IHSetUtils
│   ├── objective_functions.py
│   └── setup_spotpy.py
└── .gitignore

````

---

## :incoming_envelope: Contact us
:snake: For code-development issues contact :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria)

## :copyright: Credits
Developed by :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria).
