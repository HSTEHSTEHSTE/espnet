<!-- Generated by scripts/utils/show_asr_result.sh -->
# RESULTS
## Environments
- date: `Sat Nov 27 12:21:09 EST 2021`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.3a3`
- pytorch version: `pytorch 1.9.0`
- Git hash: `9d06e80ac454bfbc4b95575d7f2f48da0cc880f9`
  - Commit date: `Mon Nov 22 01:34:44 2021 -0500`

## asr_train_asr_raw_bpe1000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|81587|77.8|16.1|6.1|6.0|28.2|62.4|
|decode_asr_asr_model_valid.acc.ave/test|6283|40307|80.5|14.6|4.9|5.9|25.4|61.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|392279|89.7|3.9|6.4|5.7|16.0|62.4|
|decode_asr_asr_model_valid.acc.ave/test|6283|195370|91.8|3.3|4.9|5.6|13.9|61.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|115994|76.9|13.3|9.7|5.4|28.5|62.4|
|decode_asr_asr_model_valid.acc.ave/test|6283|55738|80.2|12.0|7.9|5.8|25.6|61.4|



## asr_train_asr_conformer6_raw_bpe1000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|81587|82.4|12.4|5.2|5.4|23.0|57.5|
|decode_asr_asr_model_valid.acc.ave/test|6283|40307|85.0|11.0|4.1|5.4|20.5|55.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|392279|91.6|2.9|5.4|5.3|13.7|57.5|
|decode_asr_asr_model_valid.acc.ave/test|6283|195370|93.6|2.4|4.0|5.4|11.7|55.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_all|12986|115994|81.6|10.1|8.3|5.3|23.7|57.5|
|decode_asr_asr_model_valid.acc.ave/test|6283|55738|84.9|8.6|6.5|5.7|20.7|55.5|
