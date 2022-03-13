import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


def select_negative_samples(user_item_interact: pd.DataFrame, all_items: set, alpha: int = 1) -> pd.DataFrame:
    """

    Args:
        user_item_interact: pd.DataFrame of user-item positive interactions
        all_items: set of all item_id
        alpha: the number of negative interactions in relation to number of positive interactions for each user

    Returns:
        df_negative: pd.DataFrame of user-item negative interactions

    """
    random.seed(7)
    d_positive = defaultdict(list)
    for row in user_item_interact[['user_id', 'item_id']].values:
        d_positive[row[0]].append(row[1])

    tmp = []
    for key in tqdm(d_positive):
        key_list = [int(key)] * alpha * len(d_positive[key])
        tmp.extend(tuple(zip(key_list, random.sample(all_items - set(d_positive[key]), alpha * len(d_positive[key])))))

    df_negative = pd.DataFrame(tmp, columns=['row', 'col'])
    return df_negative
