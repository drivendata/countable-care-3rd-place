# packages
APT_PKGS := python-pip python-dev
BREW_PKGS := --python
PIP_PKGS := numpy scipy pandas scikit-learn

# directories
DIR_DATA := data
DIR_BUILD := build
DIR_SRC := src
DIR_BIN := $(DIR_BUILD)/bin
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model

# directories for the cross validation and ensembling
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst

DATA_TRN := $(DIR_DATA)/train.csv
DATA_TST := $(DIR_DATA)/test.csv
LABEL_TRN := $(DIR_DATA)/label.csv

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST) $(DIR_BIN)

# data files for training and predict
SUBMISSION_SAMPLE := $(DIR_DATA)/SubmissionFormat.csv

ID_TST := $(DIR_DATA)/id.tst.txt
ID_TRN := $(DIR_DATA)/id.trn.txt
ID_VALTRN := $(DIR_DATA)/train_ids_80_percent.csv
ID_VALTST := $(DIR_DATA)/valid_ids_20_percent.csv
HEADER := $(DIR_DATA)/header.txt

YS_TRN := $(DIR_DATA)/ys.trn.csv
YS_VALTRN := $(DIR_DATA)/ys.valtrn.csv
YS_VALTST := $(DIR_DATA)/ys.valtst.csv


# initial setup
$(DIRS):
	mkdir -p $@

mac.setup: $(DIRS)
	brew install $(BREW_PKGS)
	sudo pip install $(PIP_PKGS)

ubuntu.setup: $(DIRS)
	sudo apt-get install $(APT_PKGS)
	sudo pip install $(PIP_PKGS)

$(ID_TST): $(SUBMISSION_SAMPLE)
	cut -d, -f1 $< | tail -n +2 > $@

$(ID_TRN): $(DATA_TRN)
	cut -d, -f1 $< | tail -n +2 > $@

$(HEADER): $(SUBMISSION_SAMPLE)
	head -1 $< > $@

$(YS_TRN): $(LABEL_TRN)
	cut -d, -f2- $< | tail -n +2 > $@

$(YS_VALTRN) $(YS_VALTST): $(YS_TRN) $(ID_TRN) $(ID_VALTST)
	python src/split_train_valid.py --feature-file $< \
                                    --valid-train-file $(YS_VALTRN) \
                                    --valid-test-file $(YS_VALTST) \
                                    --train-id $(word 2, $^) \
                                    --valid-test-id $(lastword $^)

# cleanup
clean::
	find . -name '*.pyc' -delete

clobber: clean
	-rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup
