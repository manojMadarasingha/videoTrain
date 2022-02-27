# videoTrain
Datasets and Related coding scripts for analysis of 360-degree video. We present only the classification with DNN models with the synthesized data because, we observe increasing accuracy in 360/normal classification only for the those models. Traditional classifiers we used, XGBoost and SVM do not show significant improvement with increasing number of data.

Note: In this repository we provide only the model for classification purpose of 360 and normal video traces. For detail implementation of the data synthesis model please refer to the extended version of **VideoTrain**, **VideoTrain++** in https://github.com/manojMadarasingha/videoTrainplusplus.

This repo includes
* raw_data:                 dataset including raw video traces for both 360 and normal video data (i.e., for both original and synthesized data)
* original_data:            processed dataset for original data used for DNN classification
* synthesized data          processed dataset for synthesized data used for DNN classification
* run_videoTrain_on_DNN.py  script for the classification 

## Brief intro fo each dataset

### raw data

Contains raw traces. Each original traces has 241 bins (note: we use only first 240 bins in this process and each bin is 0.5s long). For each bin of we have derived several statistical measures (i.e., *packet_count* (pkt_count), *frame_sum* (total bytes dl by all packets), *frame_avg* (avg. bytes dl) etc.)  for both uplink and downlink. For selected features, *pkt_count_dl* and *frame_sum_dl*, we derive synthesized traces for each trace separately which are stored in **synth** folder. In each trace, every consecutive 241 bins represent on video trace.

### original data/Synthesized data

Processed dataset for DNN model classification. Each .csv file represents YouTube/Facebook  360/Normal dataset. In each .csv file, rows represent one trace where first 240 columns represents the *frame_sum_dl* feature. Last two columns contain ground truth data *video id* and *video type* (360 = 1, Normal = 0)

## Run the classification script

### Required packages

* pandas        1.0.3
* numpy         1.18.1
* keras         2.3.1
* sklearn       0.22.1

### run script

Script requires following command line arguments

* `--t_type`                   platform type as a string var. 'YT' or 'FB' for YouTube or Facebook respectevely. Default: YT
* `--data_addition_rounds`     no. of rounds adding the synthesized data as an int var. Each iteration we add 100 synthesized traces from each category 360/Normal. Default= 10
* `--ena_only_synth_traces`     enable variable to run the classification only on synthesized data. Default: run the classification with actual+synthesized data
* `--ena_unseen`                enable variable to tun the classification with in seen condition in which traces of a given video ID is split into both train and test sets. Defulat: unseen condition is enabled where non of the traces in train video IDs are not found in test set
* `--ena_mlp`                   enable variable to run the mlp model. Default: CNN model
* `--path`                      specificy the dataset path. Default: current working directory

Output of each classifiers will be stored in a seprate folder **results** in following hierarchy

--  **results**

------  Platfom (**YouTube**/**Facebook**)

--------- Classifier type (**CNN**/**MLP**)

------------- Data split condition (**Seen**/**Unseen**)

----------------- only_synthesized_set_n.csv / ori_plus_synthesized_set_n.csv (n: no. of random train/test splits)

Sample run commands

* run classification on YouTube data adding 5 rounds of synthesized data. Conider actual + synthesized scenario in seen condition with CNN 

 `python3 run_mlp_on_row_data_copy.py --t_type 'YT' --data_addition_rounds 5`
 
* run classification on YouTube data adding 5 rounds of synthesized data. Conider synthesized scenario in seen condition with CNN 

`python3 run_mlp_on_row_data_copy.py --t_type 'YT' --data_addition_rounds 5 --ena_only_synth_traces`

* run classification on YouTube data adding 5 rounds of synthesized data. Conider synthesized scenario in unseen condition with CNN 

`python3 run_mlp_on_row_data_copy.py --t_type 'YT' --data_addition_rounds 5 --ena_only_synth_traces --ena_unseen`
 
* run classification on YouTube data adding 5 rounds of synthesized data. Conider synthesized scenario in unseen condition with mlp

 `python3 run_mlp_on_row_data_copy.py --t_type 'YT' --data_addition_rounds 5 --ena_only_synth_traces --ena_unseen --ena_mlp`
 
Citation for the related paper accepted for IEEE WoWMoM 2021.

`@inproceedings{videotrain,
  title={VideoTrain: A Generative Adversarial Framework for Synthetic Video Traffic Generation (in press)},
  author={Kattadige, Chamara and  Muramudalige, Shashika R and Choi, Kwon Nung and Jourjon, Guillaume and
  Wang, Haonan and  Jayasumana, Anura  and Thilakarathna, Kanchana},
  booktitle={22nd IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks},
  year={2021}
}`


*Chamara Kattadige, Shashika R Muramudalige, Kwon Nung Choi, Guillaume Jourjon, Haonan Wang, Anura Jayasumana, and Kanchana Thilakarathna. 2021.VideoTrain: A Generative Adversarial Framework for Synthetic Video TrafficGeneration (in press). In22nd IEEE International Symposium on a World of Wireless, Mobile and Multimedia Network*










