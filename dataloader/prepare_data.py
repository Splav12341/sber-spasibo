import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataloader.negative_samples import select_negative_samples
from dataloader.prepare_test_data import all_items_to_test_users


def filter_interactions(interactions: pd.DataFrame, threshold: int = 0) -> pd.DataFrame:
    """

    Args:
        interactions: DataFrame with interactions
        threshold: threshold for numbers of interactions for each user

    Returns:
        filtered_interactions: DataFrame with interactions where users with
            interactions less than threshold dropped

    """

    interactions_grouped = interactions[['row', 'col']].groupby('row').count().copy()
    filtered_interactions = interactions_grouped[interactions_grouped['col'] > threshold]\
        .reset_index().drop('col', axis=1)

    filtered_interactions = interactions.merge(filtered_interactions, left_on='row', right_on='row')

    return filtered_interactions


def train_val_test_split(interactions: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.05):
    """

    Args:
        interactions: DataFrame with interactions
        val_size: size of val dataset
        test_size: size of test dataset

    Returns:
        (train_interactions, val_interactions, test_interactions): tuple of pd.DataFrame
        Split interaction DataFrames by users

    """
    interactions = filter_interactions(interactions)
    train_val_users, test_users = train_test_split(interactions[['row', 'col']].groupby('row').count().reset_index(),
                                                   test_size=test_size, random_state=7)
    train_users, val_users = train_test_split(train_val_users, test_size=val_size, random_state=7)

    train_interactions = train_users.drop(['col'], axis=1).merge(interactions[['row', 'col']], left_on='row',
                                                                 right_on='row')
    val_interactions = val_users.drop(['col'], axis=1).merge(interactions[['row', 'col']], left_on='row',
                                                             right_on='row')
    test_interactions = test_users.drop(['col'], axis=1).merge(interactions[['row', 'col']], left_on='row',
                                                               right_on='row')

    return train_interactions, val_interactions, test_interactions


def get_full_dataset_from_interactions(interactions: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        interactions: pd.DataFrame with user-item positive interactions
        user_features: pd.DataFrame with user features
        item_features: pd.DataFrame with item features

    Returns:
        user_item_interact: pd.DataFrame with user-item positive interactions with
            user features with item features

    """
    user_interact = interactions[['row', 'col']].merge(user_features, left_on='row', right_on='row')
    user_item_interact = user_interact.merge(item_features, left_on='col', right_on='row')

    user_item_interact = user_item_interact.drop(['row_y'], axis=1)\
        .rename(columns={'row_x': 'user_id', 'col': 'item_id'})

    return user_item_interact


def get_all_items(item_features: pd.DataFrame) -> set:
    """

    Args:
        item_features: pd.DataFrame with item features

    Returns:
        set of all item_id

    """
    return set(list(item_features['row'].unique()))


def add_targets_and_concat(positive_dataset: pd.DataFrame, negative_dataset: pd.DataFrame = None) -> pd.DataFrame:
    """

    Args:
        positive_dataset: pd.DdtaFrame with positive interactions
        negative_dataset: pd.DdtaFrame with negative interactions

    Returns:
        dataset: pd.DdtaFrame with positive and negative interactions with targets for train and val
            or pd.DdtaFrame with positive interactions with targets for test

    """
    positive_dataset['target'] = pd.Series(np.ones(positive_dataset.shape[0]), index=positive_dataset.index)
    if type(negative_dataset) == pd.DataFrame:
        negative_dataset['target'] = pd.Series(np.zeros(negative_dataset.shape[0]), index=negative_dataset.index)
        dataset = pd.concat([positive_dataset, negative_dataset])
        return dataset
    return positive_dataset


def get_positive_and_negative_interactions(interactions: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame):
    """

    Args:
        interactions: pd.DataFrame with user-item positive interactions
        user_features: pd.DataFrame with user features
        item_features: pd.DataFrame with item features

    Returns:
        (train_data, val_data, test_data, test_user_item_interact): tuple of pd.DataFrame
            train_data: train pd.DataFrame with user-item interactions with all features with target
            val_data: val pd.DataFrame with user-item interactions with all features with target
            test_data: test pd.DataFrame with user-item interactions with all features with target
            test_user_item_interact: pd.DataFrame with user-item positive interactions with
                user features with item features
    """
    train_interactions, val_interactions, test_interactions = train_val_test_split(interactions)

    # Доюавляем к позитивным интеракциям фичи по юзеру и по айтему
    train_user_item_interact = get_full_dataset_from_interactions(train_interactions, user_features, item_features)
    val_user_item_interact = get_full_dataset_from_interactions(val_interactions, user_features, item_features)
    test_user_item_interact = get_full_dataset_from_interactions(test_interactions, user_features, item_features)

    all_items = get_all_items(item_features)

    # Получаем негативные интеракции
    alpha = 1
    train_df_negative = select_negative_samples(train_user_item_interact, all_items)
    val_df_negative = select_negative_samples(val_user_item_interact, all_items)

    # Доюавляем к негативным интеракциям фичи по юзеру и по айтему
    train_user_item_interact_negative = get_full_dataset_from_interactions(train_df_negative, user_features,
                                                                           item_features)
    val_user_item_interact_negative = get_full_dataset_from_interactions(val_df_negative, user_features, item_features)

    # Объединяем позитивные и негативные интеракции и добавляем тагрет
    train_data = add_targets_and_concat(train_user_item_interact, train_user_item_interact_negative)
    val_data = add_targets_and_concat(val_user_item_interact, val_user_item_interact_negative)

    # Готовим тестовый файл для MAP@10
    test_data = add_targets_and_concat(test_user_item_interact)
    all_items_to_user = all_items_to_test_users(test_data, all_items)
    test_user_item_interact = get_full_dataset_from_interactions(all_items_to_user, user_features, item_features)

    train_data.to_csv('./data/prepared/train_data.csv', index=False)
    val_data.to_csv('./data/prepared/val_data.csv', index=False)
    test_data.to_csv('./data/prepared/test_data.csv', index=False)
    test_user_item_interact.to_csv('./data/prepared/test_user_item_interact.csv', index=False)

    return train_data, val_data, test_data, test_user_item_interact
