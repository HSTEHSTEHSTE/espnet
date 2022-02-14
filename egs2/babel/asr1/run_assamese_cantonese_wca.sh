#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# train_set=train_cantonese
# valid_set=dev_cantonese
# test_sets=eval_cantonese
# cl_train_sets='train_assamese'
# cl_valid_sets='dev_assamese'
# cl_test_sets='eval_assamese'
train_set=train_assamese
valid_set=dev_assamese
test_sets=eval_assamese
cl_train_sets='train_cantonese'
cl_valid_sets='dev_cantonese'
cl_test_sets='eval_cantonese'



langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="107 201 307 404"

##for l in ${recog}; do
 # test_sets="dev_${l} eval_${l} ${test_sets}"
#done
#test_sets=${test_sets%% }

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decoder_asr.yaml

nlsyms_txt=data/nlsym.txt


# TODO(kamo): Derive language name from $langs and give it as --lang
./asr.sh \
    --asr_stats_dir "exp_assamese_cantonese_wca/asr_stats_raw_car" \
    --cl_asr_stats_dir "exp_assamese_cantonese_wca/cl_asr_stats_raw_car" \
    --expdir "exp_assamese_cantonese_wca" \
    --dumpdir "dump_cantonese" \
    --stage 12 \
    --stop_stage 13 \
    --lang assamese_cantonese \
    --local_data_opts "--langs ${langs} --recog ${recog}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type char \
    --feats_type raw \
    --nlsyms_txt ${nlsyms_txt} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --cl_train_sets "${cl_train_sets}" \
    --cl_valid_sets "${cl_valid_sets}" \
    --cl_test_sets "${cl_test_sets}" \
    --cl_type "wca" \
    --resume false \
    --use_ngram true \
    --lm_train_text "data/${train_set}/text" \
    --cl_lm_train_text "data/${cl_train_sets}/text" \
    --pretrained_model "/home/xli257/espnet/egs2/babel/asr1/exp_cantonese/asr_train_asr_raw_assamese_cantonese_char/valid.loss.ave_10best.pth" "$@"