import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data: pd.DataFrame) -> None:
    """Test if the DataFrame has the expected column names.
    
    Args:
        data: Input DataFrame to test
    """
    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values
    
    # This also enforces the same order
    #Ensures all required columns exist
    #Ensures the order is EXACTLY the same
    #If your new CSV adds, removes, or reorders columns → pipeline stops
    #Purpose: protect against schema drift.
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data: pd.DataFrame) -> None:
    """Test if neighborhood names are within expected values.
    
    Args:
        data: Input DataFrame to test
    """
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float) -> None:
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    
    Args:
        data: Current dataset to test
        ref_data: Reference dataset to compare against
        kl_threshold: Maximum allowed KL divergence threshold
        
    Raises:
        AssertionError: If KL divergence exceeds the threshold
    """
    # Use newer pandas value_counts with normalize=True for probability distribution
    dist1 = data['neighbourhood_group'].value_counts(normalize=True).sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts(normalize=True).sort_index()
    
    # Ensure distributions sum to 1 and have matching indices
    assert np.isclose(dist1.sum(), 1.0)
    assert np.isclose(dist2.sum(), 1.0)
    assert dist1.index.equals(dist2.index)

    # Calculate KL divergence with improved numerical stability
    kl_div = scipy.stats.entropy(dist1, dist2, base=2)
    assert np.isfinite(kl_div) and kl_div < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################
def test_row_count(data: pd.DataFrame, min_rows: int = 15000, max_rows: int = 1000000) -> None:
    """
    Ensure the dataset has a reasonable number of rows

    """
    #Number of rows must be at least 5 and not more than 1 million
    row_count = data.shape[0]

    assert row_count >= min_rows, f"Dataset has too few rows: {row_count}"
    assert row_count <= max_rows, f"Dataset has too many rows: {row_count}"

#Test price range
def test_price_range(data: pd.DataFrame, min_price: float, max_price: float) -> None:
    """
    Ensure the price column is within expected bounds.
    """ 
    prices = data['price']
    idx = prices.between(min_price, max_price)
    assert idx.all(), (
        f"Found prices outside range {min_price}–{max_price}. "
        f"Invalid prices: {prices[~idx].tolist()}"
    )

