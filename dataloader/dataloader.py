import pandas as pd


class GettingData:
    def __init__(self, dir_path: str):
        """

        Args:
            dir_path: path to directory with external data
        """
        self.dir_path = dir_path

    def _get_iteractions_data(self, filename: str):
        """

        Args:
            filename: file name of interactions

        Returns:
            interactions: pd.DataFrame with user-item positive interactions
        """

        interactions = pd.read_csv(self.dir_path + '/' + filename)
        return interactions

    @staticmethod
    def _join_items_data(item_asset: pd.DataFrame, item_price: pd.DataFrame, item_subclass: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            item_asset: pd.DataFrame with item asset info
            item_price: pd.DataFrame with item price info
            item_subclass: pd.DataFrame with item subclass info

        Returns:
            item_features: pd.DataFrame with item features

        """

        item_features_1 = item_asset[['row', 'data']].merge(item_price[['row', 'data']], left_on='row', right_on='row',
                                                            suffixes=('_asset', '_price'))
        item_features = item_features_1.merge(item_subclass[['row', 'col']], left_on='row', right_on='row').rename(
                                                        columns={'col': 'col_subclass'})

        return item_features

    def _get_items_data(self, item_filenames: list):
        """

        Args:
            item_filenames: list of str (file names with item features)

        Returns:
            item_features: pd.DataFrame with item features

        """
        item_asset = pd.read_csv(self.dir_path + '/' + item_filenames[0])
        item_price = pd.read_csv(self.dir_path + '/' + item_filenames[1])
        item_subclass = pd.read_csv(self.dir_path + '/' + item_filenames[2])

        item_features = self._join_items_data(item_asset, item_price, item_subclass)

        return item_features

    @staticmethod
    def _join_users_data(user_age: pd.DataFrame, user_region: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            user_age: pd.DataFrame with user age info
            user_region: pd.DataFrame with user region info

        Returns:
            user_features: pd.DataFrame with user features

        """
        user_features = user_region.merge(user_age, left_on='row', right_on='row',
                                          suffixes=('_region', '_age'))
        return user_features

    def _get_users_data(self, user_filenames: list) -> pd.DataFrame:
        """

        Args:
            user_filenames: list of str (file names with user features)

        Returns:
            user_features: pd.DataFrame with user features

        """
        user_age = pd.read_csv(self.dir_path + '/' + user_filenames[0])
        user_region = pd.read_csv(self.dir_path + '/' + user_filenames[1])

        user_features = self._join_users_data(user_age, user_region)

        return user_features

    def get_external_data(self, filename: str, item_filenames: list, user_filenames: list):
        """

        Args:
            filename: file name of interactions
            item_filenames: list of str (file names with item features)
            user_filenames: list of str (file names with user features)

        Returns:
            (iteractions, item_features, user_features): tuple of pd.DataFrame
                interactions: pd.DataFrame with user-item positive interactions
                user_features: pd.DataFrame with user features
                item_features: pd.DataFrame with item features

        """
        iteractions = self._get_iteractions_data(filename)
        item_features = self._get_items_data(item_filenames)
        user_features = self._get_users_data(user_filenames)

        return iteractions, item_features, user_features
