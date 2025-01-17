#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_bengali
valid_set=dev_bengali
test_sets=eval_bengali
cl_train_sets='train_assamese'
cl_valid_sets='dev_assamese'
cl_test_sets='eval_assamese'



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
    --asr_stats_dir "exp_bengali/asr_stats_raw_car" \
    --cl_asr_stats_dir "exp_bengali/cl_asr_stats_raw_car" \
    --expdir "exp_bengali" \
    --dumpdir "dump_bengali" \
    --stage 3 \
    --stop_stage 5 \
    --lang assamese_bengali \
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
    --use_ngram true \
    --lm_train_text "data/${train_set}/text" \
    --cl_lm_train_text "data/${cl_train_sets}/text" "$@"
    # --pretrained_model "/export/b13/xli/exp_2/asr_train_asr_raw_char/valid.loss.best.pth" "$@"