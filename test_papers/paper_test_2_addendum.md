# Clarifications

- In Section 3.1 Frequency-Threshold based Forcasting, the authors meant $\hat{D}_{PT}$ (the pre-training dataset, filtering out samples the model got wrong) instead of $D_{PT}$ (the entire pre-training dataset). All forecasting algorithms in Table 1 and Table 2 are evaluated on $\hat{D}_{PT}$.

- In Section 4.1 Training and Evaluation Setup, the paper describes how $D_{PT}$ is constructed, saying that they "evaluate forgetting over 36 tasks from the training split of the Public Pool of Prompts (P3) dataset" and "use a balanced sample of 100 examples per task to from our $D_{PT}$". By balanced, the paper simply means they sampled an equal number (100) of examples per task. The 36 tasks used are the intersection of T0-Train tasks (Table 5 in https://arxiv.org/pdf/2110.08207 以下是Table5中的内容一字不差地抽取出来的结果：

Table 5: All training and evaluation datasets.  
The dataset are printed in their Hugging Face datasets identifier, where the part after / is their subset name. Hotpot QA is recast as closed-book QA due to long input length. Full citations are included in Appendix G.

Task Dataset T0 Train T0+ Train T0++ Train Eval

Coreference Resolution super-glue/wsc.fixed ✓ ✓

Coreference Resolution winogrande/winogrande.xl ✓

Natural Language Inference super-glue/cb ✓

Natural Language Inference super-glue/rte ✓

Natural Language Inference anli ✓

Paraphrase Identification glue/mrpc ✓ ✓ ✓

Paraphrase Identification glue/qqp ✓ ✓ ✓

Paraphrase Identification paws/labeled-final ✓ ✓ ✓

Closed-Book QA ai2_arc/ARC_Challenge ✓ ✓

Closed-Book QA ai2_arc/ARC_Easy ✓ ✓

Closed-Book QA kilt_tasks/hotpotqa ✓ ✓ ✓

Closed-Book QA trivia_qa/unfiltered ✓ ✓

Closed-Book QA web_questions ✓ ✓

Closed-Book QA wiki_qa ✓ ✓ ✓

Extractive QA adversarial_qa/dbid ✓ ✓ ✓

Extractive QA adversarial_qa/dbert ✓ ✓ ✓

Extractive QA adversarial_qa/droberta ✓ ✓ ✓

Extractive QA duorc/SelfRC ✓ ✓ ✓

Extractive QA duorc/ParaphraseRC ✓ ✓ ✓

Extractive QA ropes ✓ ✓ ✓

Extractive QA squad_v2 ✓ ✓

Extractive QA super_glue/record ✓ ✓

Extractive QA quoref ✓ ✓ ✓

Extractive QA tydiqa ✓ ✓ ✓

Multiple-Choice QA cos_e/v1.11 ✓ ✓ ✓

Multiple-Choice QA cosmos_qa ✓ ✓ ✓

Multiple-Choice QA dream ✓ ✓ ✓

Multiple-Choice QA openbookqa/main ✓

Multiple-Choice QA qasc ✓ ✓ ✓

Multiple-Choice QA quail ✓ ✓ ✓

Multiple-Choice QA quarel ✓ ✓ ✓

Multiple-Choice QA quartz ✓ ✓

Multiple-Choice QA race/high ✓

Multiple-Choice QA race/middle ✓

Multiple-Choice QA sciq ✓ ✓

Multiple-Choice QA sociali_qa

Multiple-Choice QA super_glue/boolq

Multiple-Choice QA super_glue/multirc

Multiple-Choice QA wiki_hop/original ✓ ✓ ✓

Multiple-Choice QA wiqa ✓ ✓ ✓

Multiple-Choice QA piqa ✓ ✓ ✓

Sentiment amazon_polarity ✓ ✓ ✓

Sentiment app_reviews ✓ ✓ ✓

Sentiment imdb ✓ ✓ ✓

Sentiment rotten_tomatoes ✓ ✓ ✓

Sentiment yelp_review_full ✓ ✓ ✓

Sentence Completion super_glue/copa ✓

Sentence Completion story_cloze/2016 ✓

Sentence Completion hellaswag ✓ ✓ ✓

Structure-to-Text common_gen ✓ ✓ ✓

Structure-to-Text wiki_bio ✓ ✓ ✓

Summarization cnn_dailymail/3.0.0 ✓ ✓ ✓

Summarization gigaword ✓ ✓ ✓

Summarization multi_news ✓ ✓ ✓

Summarization samsum ✓ ✓ ✓

Summarization xsum ✓ ✓ ✓

Topic Classification ag_news ✓ ✓ ✓

Topic Classification dbpedia_14 ✓ ✓ ✓

Topic Classification trec ✓ ✓ ✓

Word Sense Disambiguation super_glue/wic ✓ ✓

--- 

以上内容完全按照Table5中的表格和注释逐字提取，未作任何修改或删减。) and the tasks in used in the repo of BART0 model (https://github.com/INK-USC/ReCross/blob/main/data/):
    - glue-mrpc
    - glue-qqp
    - paws_x-en
    - kilt_tasks-hotpotqa
    - wiki_qa
    - adversarial_qa-dbert
    - adversarial_qa-dbidaf
    - adversarial_qa-droberta
    - duorc-SelfRC
    - duorc-ParaphraseRC
    - ropes
    - quoref
    - cos_e-v1.11
    - cosmos_qa
    - dream
    - qasc
    - quail
    - quartz
    - sciq
    - social_i_qa
    - wiki_hop-original
    - wiqa
    - amazon_polarity
    - app_reviews
    - imdb
    - rotten_tomatoes
    - yelp_review_full
    - common_gen
    - wiki_bio
    - cnn_dailymail-3.0.0
    - gigaword
    - multi_news
    - samsum
    - xsum
    - ag_news
    - dbpedia_14

- In Section 4.1 Tasks for Model Refinement, the paper says that MMLU is used to refine FLAN-T5, however the specific split isn't specified. The validation set is used, specifically the one from the original release of MMLU (https://people.eecs.berkeley.edu/~hendrycks/data.tar).

- In Section 4.1 Training and Evaluation of the Forecasting Model, the paper says that they "collect mis-predicted examples from the training split $D^{Train}_R$ and the test split $D^{Test}_R$", however the process by which they're collected isn't specified. First, $D_R$ is constructed by evaluating the pretrained LMs on each new task and keeping only those examples where model predictions are wrong. Here, a prediction is graded using the exact match metric using the evaluation script of SQuAD 2.0 (https://rajpurkar.github.io/SQuAD-explorer/ 网页内容为：What is SQuAD?
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

SQuAD 1.1, the previous version of the SQuAD dataset, contains 100,000+ question-answer pairs on 500+ articles.

Getting Started
We've built a few resources to help you get started with the dataset.

Download a copy of the dataset (distributed under the CC BY-SA 4.0 license):

To evaluate your models, we have also made available the evaluation script we will use for official evaluation, along with a sample prediction file that the script will take as input. To run the evaluation, use python evaluate-v2.0.py <path_to_dev-v2.0> <path_to_predictions>.

Once you have a built a model that works to your expectations on the dev set, you submit it to get official scores on the dev and a hidden test set. To preserve the integrity of test results, we do not release the test set to the public. Instead, we require you to submit your model so that we can run it on the test set for you. Here's a tutorial walking you through official evaluation of your model:

Because SQuAD is an ongoing effort, we expect the dataset to evolve.

To keep up to date with major changes to the dataset, please subscribe:). Second, $D_R$ is randomly split into 60% and 40% subsets to create $D^{Train}_R$ and $D^{Test}_R$ respectively. More details about $D_R$: For T5 experiments, validation split of 57 tasks from MMLU was used. For BART0 experiments, the test split in https://github.com/INK-USC/ReCross/blob/main/data/ was used, and the set of test tasks used were:
    - super_glue-wsc.fixed
    - winogrande-winogrande_xl
    - super_glue-cb
    - super_glue-rte
    - anli
    - super_glue-copa
    - hellaswag
    - super_glue-wic

- There are no specifics as for how LoRA was applied to FLAN-T5_{Large} in Section 4. LoRA was applied to the query and value matrices in all self-attention layers:

    ```python
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, bias="none", 
        target_modules=['q', 'v'],
    )
    ```

- In Appendix B, the paper says that the P3 dataset has been split into in- and out-of-distribution splits by task (e.g. `anli` is in the in-distribution split). There are many variants of each task in the P3 dataset, though (e.g. for the `anli` task, there's `anli_claim_true_false_inconclusive_r1_score_eval`, `anli_claim_true_false_inconclusive_r1`, etc) but it's not specified whether all variants are used, or just a subset. All task variants are used.

- The full fine-tuning setup was used to produce Table 2, **not** head-only or LoRA.

- The hyperparameters specified in Section 4.1 -- Hyperparameters were used to produce Figure 3.

- When fine-tuning models to produce Figure 3, each example $(x_i, y_i) \in D_R^{test}$ is sampled at random. That is, the examples in each data stream were randomly shuffled.

- In Figure 3, is the forecasted forgetting binary indicator computed by $g$ at the start and used for every time step (i.e. $\hat{z}_{ij}$) or is it computed each timestep (i.e. $\hat{z}_{ij}^{t}$)? The forgetting is computed at the start and used for every time step.


# Out of Scope

- In Table 3, results from other papers are mentioned (MIR and OCS); these are out of scope.

- The "Hyperparameter Analysis" sub-section in Section 5.2, Section 5.3 and Table 5 are all out of scope.
