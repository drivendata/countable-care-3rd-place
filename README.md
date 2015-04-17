Countable Care
==============

## Requirements
### Python Packages
Install python packages listed in `requirements.txt` - `scipy`, `numpy`,
`scikit-learn`, `statsmodels`, `pandas`, `Kaggler` packages

### XGBoost 0.3
Install latest `XGBoost` from source and copy `xgboost` and
`wrapper/libxgboostwrapper.so` into the system `bin` and `lib` folders
respectively:
```
git clone git@github.com:dmlc/xgboost.git
cd xgboost
bash build.sh
(sudo) cp xgboost /usr/local/bin
(sudo) cp wrapper/libxgboostwrapper.so /usr/local/lib
```

### Kaggler 0.3.8
To install latest `Kaggler` package from source:
```
git clone git@github.com:jeongyoonlee/Kaggler.git
cd Kaggler
python setup.py build_ext --inplace
(sudo) python setup.py install
```

## Features
8 features are used as follows:
* `feature1` - impute 0 for missing values for numeric and ordinal features.
* create dummy variables for values in categorical features appearing 10+ times
* in training data
* `feature2` - same as `feature1` except taking `log(1 + x)` transformation for
* ordinal features.
* `feature3` - same as `feature2` except creating dummy variables for values
* appearing 3+ times in training data.
* `feature4` - same as `feature3` except treating ordinal features as
* categorical features.
* `feature5` - same as `feature4` except taking `log2(1 + x)` transformation
* for ordinal features before treating ordinal features as categorical
* features.
* `feature8` - same as `feature4` except normalizing numeric features.
* `feature9` - impute -1 for missing values, and label-encode categorical
* features.
* `feature10` - impute 0 for missing values for numeric features, and
* label-encode categorical features.

## How to Generate Features
You can generate feature files manually using relevant Makefiles.  For example,
to generate `feature1` files for class `00` out of 14 classes:
```
make -f Makefile.feature.feature1 build/feature/feature1.trn00.sps
```

or you can run an algorithm Makefile that uses `featuer1`, then feature files
will be generated automatically before training:
```
make -f Makefile.xg_100_8_0.05_feature1
```

## Algorithm Implementations
6 different algorithm implementations are used as follows:
* `fm` - Factorization Machine implementation from
* [Kagger](https://github.com/jeongyoonlee/Kaggler)
* `nn` - Neural Networks implementation from
* [Kaggler](https://github.com/jeongyoonlee/Kaggler)
* `lr` - Logistic Regression implementation from
* [Scikit-Learn](http://scikit-learn.org/stable/)
* `gbm` - Gradient Boosting Machine implementation from
* [Scikit-Learn](http://scikit-learn.org/stable/)
* `libfm` - Factorization Machine implementation from
* [libFM](http://www.libfm.org/)
* `xg` - Gradient Boosting Machine implementation from
* [XGBoost](https://github.com/dmlc/xgboost)

## Individual Models
From 6 different algorithm implementations and 8 different features (see
[Features](features)), 19 individual models are built as follows:
* `fm_200_8_0.001_feature2`
* `fm_200_8_0.001_feature2`
* `fm_200_8_0.001_feature3`
* `gbm_bagging_40_7_0.1_feature10`
* `libfm_200_4_0.005_feature2`
* `libfm_200_4_0.005_feature4`
* `lr_0.1_feature2`
* `lr_0.1_feature4`
* `nn_20_64_0.005_feature8`
* `nn_20_8_0.01_feature2`
* `nn_20_8_0.01_feature3`
* `rf_400_40_feature2`
* `rf_400_40_feature5`
* `rf_400_40_feature9`
* `rf_400_40_feature10`
* `xg_100_8_0.05_feature1`
* `xg_100_8_0.05_feature5`
* `xg_100_8_0.05_feature8`
* `xg_100_8_0.05_feature9`
* `xg_100_8_0.05_feature10`
* `xg_bagging_120_7_0.1_feature9`

## How to Generate Individual Model Predictions
Each model has its Makefile available for training and prediction.  For
example, to generate predictions for `fm_200_8_0.001_feature2`, run:
```
make -f fm_200_8_0.001_feature2
```

Predictions for training data with 5-CV and test data will be saved in
`build/val` and `build/tst` folders respectively.

## Ensemble Model
Using predictions of 19 individual models (see [Individual Models](individual
models)) as inputs, a Gradient Boosting Machine ensemble model is trained as
follows:
* `esb_xg_grid_colsub`

Parameters for the ensemble model are selected for each class by using grid
search.

## How to Generate Ensemble Prediction
After generating individual model predictions, run the ensemble Makefile as
follows:
```
make -f Makefile.esb.xg_grid_colsub
```

The prediction and submission files will be available in the `build/tst`
folder.
