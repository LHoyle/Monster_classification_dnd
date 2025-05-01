#!/bin/bash
feature=['type','size','ac','hp','speed','align','legendary','source','str','dex','con','int','wis','cha']
label=cr

# action choices
#  choices=[ "fit", "score", "loss", "cross", "predict", "grid-search", "show-best-params", "random-search",
# "cross-score", "confusion-matrix", "precision-recall-plot", "pr-curve" ]
# 

# model choices
# choices=["SGD", "linear", "SVM", "boost", "forest", "tree"]
# choices=[]

model_type=boost
miss_strat_num=""
miss_strat_cat=""
train=monsters_redux_electric_boogaloo-train.csv
test=monsters_redux-verification.csv
label=cr
model=model_1.joblib
seed=1073198745193856139486
poly_deg=2
scalar=1
show_test=1
cv_count=100
n_search=100
image_file=Model1.png
# python3 display_data.py all -d monsters_redux.csv --label  cr

# python3 split_data.py -d midmonsters_redux.csv -t monsters_redux_electric_boogaloo-train.csv -T monsters_redux_electric_boogaloo-test.csv -v monsters_redux-verification.csv


# python3 pipeline.py cross -M ${model_type} -t ${train} -T ${test} -m ${model} -R ${seed} -l ${label} -p ${poly_deg} -S ${show_test}  -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}
# # python3 pipeline.py random-search -M ${model} -t ${train} -T ${test} -m ${model} R ${seed} -l ${model} -p ${poly_deg} -S ${show_test} -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}

# python3 pipeline.py loss -M ${model_type} -t ${train} -T ${test}  -m ${model} -R ${seed} -l ${label} -p ${poly_deg} -S ${show_test}  -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}
# python3 pipeline.py cross-score -M ${model_type} -t ${train} -T ${test} -m ${model} -R ${seed} -l ${label} -p ${poly_deg} -S ${show_test}  -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}

filename=cv-${cv_count}-n_search-${n_search}-poly-${poly_deg}.txt
for choice in SGD linear SVM boost forest tree; do
echo ${choice} >>out1-${filename}
echo ${choice} >>out2-${filename}

python3 pipeline.py cross -M ${choice} -t ${train} -T ${test} -m ${model} -R ${seed} -l ${label} -p ${poly_deg} -S ${show_test}  -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}>>out1-${filename}
# python3 pipeline.py random-search -M ${model} -t ${train} -T ${test} -m ${model} R ${seed} -l ${model} -p ${poly_deg} -S ${show_test} -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}

# python3 pipeline.py loss -M ${choice} -t ${train} -T ${test}  -m ${model} -R ${seed} -l ${label} -p ${poly_deg} -S ${show_test}  -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}
python3 pipeline.py cross-score -M ${choice} -t ${train} -T ${test} -m ${model} -R ${seed} -l ${label} -p ${poly_deg} -S ${show_test}  -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}>>out2-${filename}
# done


# python3 pipeline.py -M ${model_type} -t ${train} -T ${test} -m ${model} -R ${seed} -l ${label} -p ${poly_deg} -S ${show_test} --categorical-missing-strategy ${miss_strat_cat} --numerical-missing-strategy ${miss_strat_num} -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}
# # python3 pipeline.py random-search -M ${model} -t ${train} -T ${test} -m ${model} R ${seed} -l ${model} -p ${poly_deg} -S ${show_test} --categorical-missing-strategy ${miss_strat_cat}--numerical-missing-strategy ${miss_strat_num} -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}

# python3 pipeline.py loss -M ${model_type} -t ${train} -T ${test}  -m ${model} R ${seed} -l ${label} -p ${poly_deg} -S ${show_test} --categorical-missing-strategy ${miss_strat_cat} --numerical-missing-strategy ${miss_strat_num} -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}
# python3 pipeline.py score -M ${model_type} -t ${train} -T ${test} -m ${model} R ${seed} -l ${label} -p ${poly_deg} -S ${show_test} --categorical-missing-strategy ${miss_strat_cat} --numerical-missing-strategy ${miss_strat_num} -s ${scalar} --n-search-iterations ${n_search} --cv-count ${cv_count} --image-file ${image_file}

# python3 display_data.py all -d middnd_monsters.csv -f ${feature} -l ${label}

# python3 display_data.py feature-histograms -d middnd_monsters.csv -f ${feature} -l ${label}