# On the Automatic Generation of Medical Imaging Reports
 A `pytorch` implementation of `On the Automatic Generation of Medical Imaging Reports`.

The detail of the paper can be found in [On the Automatic Generation of Medical Imaging Reports](https://arxiv.org/abs/1711.08195).


## Performance

From model only_training/only_training/20180528-02:44:52/

| Mode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDEr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 0.386 | 0.275 | 0.215 | 0.176 | 0.187 | 0.369 | 1.075 |
| Val | 0.303 | 0.182 | 0.118 | 0.077 | 0.143 | 0.256 | 0.214 |
| Test | 0.316 | 0.190 | 0.123 | 0.081 | 0.148 | 0.264 | 0.221 |
| Paper | 0.517 | 0.386 | 0.306 | 0.247 | 0.217 | 0.447 | 0.327 |

### Tags Prediction

![Stary 2018-07-07 at 10.30.57 AM](http://o7d2h0gjo.bkt.clouddn.com/2018-07-07-Stary%202018-07-07%20at%2010.30.57%20AM.png)

### Comparison

![Stary 2018-07-07 at 10.31.02 AM](http://o7d2h0gjo.bkt.clouddn.com/2018-07-07-Stary%202018-07-07%20at%2010.31.02%20AM.png)


## Visual Results

![Stary 2018-07-07 at 10.26.54 AM](http://o7d2h0gjo.bkt.clouddn.com/2018-07-07-Stary%202018-07-07%20at%2010.26.54%20AM.png)

![Stary 2018-07-07 at 10.26.30 AM](http://o7d2h0gjo.bkt.clouddn.com/2018-07-07-Stary%202018-07-07%20at%2010.26.30%20AM.png)

![Stary 2018-07-07 at 10.26.37 AM](http://o7d2h0gjo.bkt.clouddn.com/2018-07-07-Stary%202018-07-07%20at%2010.26.37%20AM.png)

![Stary 2018-07-07 at 10.26.45 AM](http://o7d2h0gjo.bkt.clouddn.com/2018-07-07-Stary%202018-07-07%20at%2010.26.45%20AM.png)


## Training


```
usage: trainer.py [-h] [--patience PATIENCE] [--mode MODE]
                  [--vocab_path VOCAB_PATH] [--image_dir IMAGE_DIR]
                  [--caption_json CAPTION_JSON]
                  [--train_file_list TRAIN_FILE_LIST]
                  [--val_file_list VAL_FILE_LIST] [--resize RESIZE]
                  [--crop_size CROP_SIZE] [--model_path MODEL_PATH]
                  [--load_model_path LOAD_MODEL_PATH]
                  [--saved_model_name SAVED_MODEL_NAME] [--momentum MOMENTUM]
                  [--visual_model_name VISUAL_MODEL_NAME] [--pretrained]
                  [--classes CLASSES]
                  [--sementic_features_dim SEMENTIC_FEATURES_DIM] [--k K]
                  [--attention_version ATTENTION_VERSION]
                  [--embed_size EMBED_SIZE] [--hidden_size HIDDEN_SIZE]
                  [--sent_version SENT_VERSION]
                  [--sentence_num_layers SENTENCE_NUM_LAYERS]
                  [--dropout DROPOUT] [--word_num_layers WORD_NUM_LAYERS]
                  [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                  [--epochs EPOCHS] [--clip CLIP] [--s_max S_MAX]
                  [--n_max N_MAX] [--lambda_tag LAMBDA_TAG]
                  [--lambda_stop LAMBDA_STOP] [--lambda_word LAMBDA_WORD]

optional arguments:
  -h, --help            show this help message and exit
  --patience PATIENCE
  --mode MODE
  --vocab_path VOCAB_PATH
                        the path for vocabulary object
  --image_dir IMAGE_DIR
                        the path for images
  --caption_json CAPTION_JSON
                        path for captions
  --train_file_list TRAIN_FILE_LIST
                        the train array
  --val_file_list VAL_FILE_LIST
                        the val array
  --resize RESIZE       size for resizing images
  --crop_size CROP_SIZE
                        size for randomly cropping images
  --model_path MODEL_PATH
                        path for saving trained models
  --load_model_path LOAD_MODEL_PATH
                        The path of loaded model
  --saved_model_name SAVED_MODEL_NAME
                        The name of saved model
  --momentum MOMENTUM
  --visual_model_name VISUAL_MODEL_NAME
                        CNN model name
  --pretrained          not using pretrained model when training
  --classes CLASSES
  --sementic_features_dim SEMENTIC_FEATURES_DIM
  --k K
  --attention_version ATTENTION_VERSION
  --embed_size EMBED_SIZE
  --hidden_size HIDDEN_SIZE
  --sent_version SENT_VERSION
  --sentence_num_layers SENTENCE_NUM_LAYERS
  --dropout DROPOUT
  --word_num_layers WORD_NUM_LAYERS
  --batch_size BATCH_SIZE
  --learning_rate LEARNING_RATE
  --epochs EPOCHS
  --clip CLIP           gradient clip, -1 means no clip (default: 0.35)
  --s_max S_MAX
  --n_max N_MAX
  --lambda_tag LAMBDA_TAG
  --lambda_stop LAMBDA_STOP
  --lambda_word LAMBDA_WORD
```


## Tester

```
usage: tester.py [-h] [--model_dir MODEL_DIR] [--image_dir IMAGE_DIR]
                 [--caption_json CAPTION_JSON] [--vocab_path VOCAB_PATH]
                 [--file_lits FILE_LITS] [--load_model_path LOAD_MODEL_PATH]
                 [--resize RESIZE] [--cam_size CAM_SIZE]
                 [--generate_dir GENERATE_DIR] [--result_path RESULT_PATH]
                 [--result_name RESULT_NAME] [--momentum MOMENTUM]
                 [--visual_model_name VISUAL_MODEL_NAME] [--pretrained]
                 [--classes CLASSES]
                 [--sementic_features_dim SEMENTIC_FEATURES_DIM] [--k K]
                 [--attention_version ATTENTION_VERSION]
                 [--embed_size EMBED_SIZE] [--hidden_size HIDDEN_SIZE]
                 [--sent_version SENT_VERSION]
                 [--sentence_num_layers SENTENCE_NUM_LAYERS]
                 [--dropout DROPOUT] [--word_num_layers WORD_NUM_LAYERS]
                 [--s_max S_MAX] [--n_max N_MAX] [--batch_size BATCH_SIZE]
                 [--lambda_tag LAMBDA_TAG] [--lambda_stop LAMBDA_STOP]
                 [--lambda_word LAMBDA_WORD]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
  --image_dir IMAGE_DIR
                        the path for images
  --caption_json CAPTION_JSON
                        path for captions
  --vocab_path VOCAB_PATH
                        the path for vocabulary object
  --file_lits FILE_LITS
                        the path for test file list
  --load_model_path LOAD_MODEL_PATH
                        The path of loaded model
  --resize RESIZE       size for resizing images
  --cam_size CAM_SIZE
  --generate_dir GENERATE_DIR
  --result_path RESULT_PATH
                        the path for storing results
  --result_name RESULT_NAME
                        the name of results
  --momentum MOMENTUM
  --visual_model_name VISUAL_MODEL_NAME
                        CNN model name
  --pretrained          not using pretrained model when training
  --classes CLASSES
  --sementic_features_dim SEMENTIC_FEATURES_DIM
  --k K
  --attention_version ATTENTION_VERSION
  --embed_size EMBED_SIZE
  --hidden_size HIDDEN_SIZE
  --sent_version SENT_VERSION
  --sentence_num_layers SENTENCE_NUM_LAYERS
  --dropout DROPOUT
  --word_num_layers WORD_NUM_LAYERS
  --s_max S_MAX
  --n_max N_MAX
  --batch_size BATCH_SIZE
  --lambda_tag LAMBDA_TAG
  --lambda_stop LAMBDA_STOP
  --lambda_word LAMBDA_WORD
```


### Method:

* test(): Compute loss
* generate(): generate captions for each image, and saved result (json) in `os.path.join(model_dir, result_path)`.
* sample(img_name): generate a caption for an image and its heatmap (`cam`).

## quantify the model performance

```
python2 metric_performance.py
```

```
usage: metric_performance.py [-h] [--result_path RESULT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --result_path RESULT_PATH
```


## Review generated captions

By using jupyter to read `review_captions.ipynb`, you can review the model generated captions for each image.

## visualize training procedure
By changing `tensorboard --logdir report_models`  to your owned saved models path in tensorboard.sh, you can visualize training procedure.

```
./tensorboard.sh
```

## Improve performance by change the model
In `utils/models`, I have implemented all models in basic version, and I think there will be some more powerful model structures which can improve the performance. So enjoy your work `^_^`.



