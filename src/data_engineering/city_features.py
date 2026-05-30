"""City and university feature engineering: geocoding, fuzzy matching, and startup hub classification."""
import pandas as pd
from rapidfuzz import process, fuzz


top_us_institution_city_map = {
    "Harvard University": "Cambridge, MA",
    "Harvard": "Cambridge, MA",
    "Stanford University": "Stanford",
    "Stanford": "Stanford",
    "Massachusetts Institute of Technology": "Cambridge, MA",
    "MIT": "Cambridge, MA",
    "University of California Berkeley": "Berkeley",
    "UC Berkeley": "Berkeley",
    "California Institute of Technology": "Pasadena",
    "University of Chicago": "Chicago",
    "Princeton University": "Princeton",
    "Yale University": "New Haven",
    "Yale": "New Haven",
    "Columbia University": "New York",
    "University of Pennsylvania": "Philadelphia",
    "University of California Los Angeles": "Los Angeles",
    "Cornell University": "Ithaca",
    "University of Michigan": "Ann Arbor",
    "Duke University": "Durham",
    "Northwestern University": "Evanston",
    "University of California San Diego": "La Jolla",
    "Brown University": "Providence",
    "Johns Hopkins University": "Baltimore",
    "University of Southern California": "Los Angeles",
    "New York University": "New York",
    "University of Wisconsin Madison": "Madison",
    "University of North Carolina at Chapel Hill": "Chapel Hill",
    "University of Texas at Austin": "Austin",
    "Carnegie Mellon University": "Pittsburgh",
    "University of Washington": "Seattle",
    "Emory University": "Atlanta",
    "Vanderbilt University": "Nashville",
    "Rice University": "Houston",
    "Georgetown University": "Washington",
    "University of Florida": "Gainesville",
    "University of California Davis": "Davis",
    "University of California Irvine": "Irvine",
    "University of Rochester": "Rochester",
    "Tufts University": "Medford",
    "University of Notre Dame": "Notre Dame",
    "Wake Forest University": "Winston-Salem",
    "Boston University": "Boston",
    "University of Virginia": "Charlottesville",
    "University of Illinois Urbana-Champaign": "Champaign",
    "Washington University in St. Louis": "St. Louis",
    "Case Western Reserve University": "Cleveland",
    "Georgia Institute of Technology": "Atlanta",
    "University of Miami": "Coral Gables",
    "University of Pittsburgh": "Pittsburgh",
    "University of California Santa Barbara": "Santa Barbara",
    "University of Minnesota Twin Cities": "Minneapolis",
    "Pennsylvania State University": "University Park",
    "University of Colorado Boulder": "Boulder",
    "Boston College": "Chestnut Hill",
}

europe_top_unis = {
    "ETH Zurich": "Zurich",
    "ETH": "Zurich",
    "Imperial College London": "London",
    "University of Oxford": "Oxford",
    "Oxford": "Oxford",
    "University of Cambridge": "Cambridge",
    "Cambridge": "Cambridge",
    "UCL": "London",
    "University of Edinburgh": "Edinburgh",
    "University of Manchester": "Manchester",
    "King's College London": "London",
    "Université PSL": "Paris",
    "EPFL": "Lausanne",
    "Technical University of Munich": "Munich",
    "London School of Economics and Political Science": "London",
    "Lund University": "Lund",
    "University of Bristol": "Bristol",
    "Delft University of Technology": "Delft",
    "University of Glasgow": "Glasgow",
    "University of Amsterdam": "Amsterdam",
    "Heidelberg University": "Heidelberg",
    "Ludwig Maximilian University of Munich": "Munich",
    "University of Leeds": "Leeds",
    "University of Warwick": "Coventry",
    "Ruprecht-Karls-Universität Heidelberg": "Heidelberg",
    "Karolinska Institute": "Stockholm",
    "KU Leuven": "Leuven",
    "École Normale Supérieure": "Paris",
    "University College Dublin": "Dublin",
    "Trinity College Dublin": "Dublin",
    "University of Copenhagen": "Copenhagen",
    "Sorbonne University": "Paris",
    "Paris Saclay University": "Paris",
    "Utrecht University": "Utrecht",
    "Leiden University": "Leiden",
    "RWTH Aachen University": "Aachen",
    "TUM": "Munich",
    "RWTH": "Aachen",
    "KIT": "Karslruhe",
    "HPI": "Potsdam",
    "Technical University of Denmark": "Lyngby",
    "University of Bonn": "Bonn",
    "University of Vienna": "Vienna",
    "University of Zurich": "Zurich",
    "Heidelberg University": "Heidelberg",
    "University of Groningen": "Groningen",
    "University of Barcelona": "Barcelona",
    "University of Helsinki": "Helsinki",
    "University of Edinburgh": "Edinburgh",
    "University of Glasgow": "Glasgow",
    "University of Milan": "Milan",
    "University of Bologna": "Bologna",
    "University of Oslo": "Oslo",
    "University of Stockholm": "Stockholm",
    "University of Warsaw": "Warsaw",
    "University of Antwerp": "Antwerp",
    "University of Zurich": "Zurich",
}

asia_top_50 = {
    "Peking University": "Beijing",
    "The University of Hong Kong": "Hong Kong",
    "National University of Singapore": "Singapore",
    "Nanyang Technological University": "Singapore",
    "Fudan University": "Shanghai",
    "The Chinese University of Hong Kong": "Hong Kong",
    "Tsinghua University": "Beijing",
    "Zhejiang University": "Hangzhou",
    "Yonsei University": "Seoul",
    "City University of Hong Kong": "Hong Kong",
    "HKUST": "Hong Kong",
    "Universiti Malaya": "Kuala Lumpur",
    "Korea University": "Seoul",
    "Shanghai Jiao Tong University": "Shanghai",
    "KAIST": "Daejeon",
    "Sungkyunkwan University": "Seoul",
    "The Hong Kong Polytechnic University": "Hong Kong",
    "Seoul National University": "Seoul",
    "Hanyang University": "Seoul",
    "Universiti Putra Malaysia": "Serdang",
    "University of Tokyo": "Tokyo",
    "University of Delhi": "New Delhi",
    "Indian Institute of Technology Delhi": "New Delhi",
    "Indian Institute of Technology Bombay": "Mumbai",
    "University of Hong Kong Polytechnic": "Hong Kong",  # alt
    "University of Malaya": "Kuala Lumpur",
    "Chulalongkorn University": "Bangkok",
    "University of Indonesia": "Depok",
    "University of Science Malaysia": "Penang",
    "Gadjah Mada University": "Yogyakarta",
    "Universitas Indonesia": "Depok",
    "National Taiwan University": "Taipei",
    "University of the Philippines": "Quezon City",
    "Ateneo de Manila University": "Quezon City",
    "University of Tehran": "Tehran",
    "Sharif University of Technology": "Tehran",
    "University of Tokyo": "Tokyo",  # THE ranking inclusion
    "University of Kyoto": "Kyoto",
    "Osaka University": "Osaka",
    "Kyoto University": "Kyoto",
    "National Tsing Hua University": "Hsinchu",
    "National Chiao Tung University": "Hsinchu",
    "Yonsei University": "Seoul",
    "Korea Advanced Institute of Science and Technology": "Daejeon",  # KAIST
    "Seoul National University": "Seoul",
    "University of Malaya": "Kuala Lumpur",
    "Tel Aviv University": "Tel Aviv",
    "Khalifa University": "Abu Dhabi",
    "Nazarbayev University": "Nur-Sultan",
    "Quaid-i-Azam University": "Islamabad",
}


# Step 1: Load CSV with no headers
institution_city_df = pd.read_csv(
    "src/notebooks/institution_city_map.csv",
    header=None,
    names=["university_name", "city"],
)

# Step 2: Create base mapping
csv_city_map = dict(
    zip(institution_city_df["university_name"], institution_city_df["city"])
)

# Step 3: Merge with hardcoded mappings
combined_uni_city_map = {
    **csv_city_map,
    **top_us_institution_city_map,
    **europe_top_unis,
    **asia_top_50,
}


# Step 4: Fuzzy match function
def fuzzy_match_city(name, mapping_dict, scorer=fuzz.token_sort_ratio, threshold=85):
    if pd.isna(name):
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    if not mapping_dict:
        return None

    result = process.extractOne(name, mapping_dict.keys(), scorer=scorer)
    if result is None:
        return None

    match, score = result[0], result[1]
    return mapping_dict[match] if score >= threshold else None


# Step 5: Apply to DataFrame
def create_university_cities(university_nodes):
    university_nodes["city"] = university_nodes["university_name"].apply(
        lambda name: fuzzy_match_city(name, combined_uni_city_map)
    )

    return university_nodes


# Step 6: Run
# df = pd.read_csv("/home/cedric-lorenz/Documents/Uni/MA/Code/gnn-startup-success/data/graph/university_nodes.csv")
# university_nodes = create_university_cities(df)
# university_nodes.to_csv(
#     "/home/cedric-lorenz/Documents/Uni/MA/Code/gnn-startup-success/data/graph/university_nodes.csv",
#     index=False
# )
