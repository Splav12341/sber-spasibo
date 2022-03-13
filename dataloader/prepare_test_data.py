import pandas as pd
from tqdm import tqdm


def all_items_to_test_users(test_data: pd.DataFrame, all_items) -> pd.DataFrame:
    """

    Args:
        test_data: pd.DdtaFrame with positive interactions with targets
        all_items: set of all item_id

    Returns:
        all_items_to_user: pd.DataFrame with all possible interactions for test users

    """

    test_users = test_data.groupby('user_id').count().reset_index()['user_id']

    tmp = []
    for user in tqdm(test_users):
        for item in all_items:
            tmp.append((user, item))

    all_items_to_user = pd.DataFrame(tmp, columns=['row', 'col'])

    return all_items_to_user
