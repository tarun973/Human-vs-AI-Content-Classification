from datasets import load_dataset
import pandas as pd

from utils import preprocess_text, sampler, gru_runner, init_nltk, tcn_runner, transformer_runner

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    ds = load_dataset("artem9k/ai-text-detection-pile")
    init_nltk()
    raw_train = ds["train"].to_pandas() # Loading only train because that is the only split available

    sampled_df = sampler(raw_train, sample_size=1000)
    out_of_fold = sampler(raw_train, sample_size=50)

    preprocessed = preprocess_text(sampled_df)

    encoding = {"ai": 1, "human": 0}
    preprocessed["source"] = preprocessed["source"].map(encoding)

    transformer_runner(sampled_df["text"].tolist(), sampled_df["source"])
    tcn_runner(preprocessed)
    gru_runner(preprocessed, out_of_fold)



