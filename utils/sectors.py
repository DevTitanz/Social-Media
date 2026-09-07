COMPANY_SECTORS = {
    # Indian Conglomerates
    "reliance": ["Energy", "Retail", "Telecom", "Media", "Finance"],
    "tata": ["Automotive", "IT", "Steel", "Consumer", "Finance"],
    "adani": ["Energy", "Infrastructure", "Ports", "Agriculture", "Media"],
    "mahindra": ["Automotive", "IT", "Finance", "Agriculture", "Real Estate"],
    "birla": ["Cement", "Telecom", "Finance", "Fashion", "Metals"],
    "bajaj": ["Automotive", "Finance", "Insurance", "Consumer"],
    "wipro": ["IT", "Healthcare", "Consumer", "Infrastructure"],
    "infosys": ["IT", "Finance", "Retail", "Manufacturing"],
    "tcs": ["IT", "Finance", "Retail", "Healthcare"],
    "hdfc": ["Finance", "Banking", "Insurance", "Real Estate"],
    "icici": ["Banking", "Finance", "Insurance", "Investment"],
    "sbi": ["Banking", "Finance", "Insurance", "Investment"],
    "ongc": ["Energy", "Oil", "Gas", "Petrochemicals"],
    "ntpc": ["Energy", "Power", "Renewables"],
    "itc": ["FMCG", "Hotels", "Agriculture", "Packaging", "Paper"],
    "hindustan unilever": ["FMCG", "Consumer", "Beauty", "Home Care"],
    "hul": ["FMCG", "Consumer", "Beauty", "Home Care"],
    "sun pharma": ["Pharmaceuticals", "Healthcare", "Specialty Chemicals"],
    "drreddy": ["Pharmaceuticals", "Healthcare", "Generics"],
    "cipla": ["Pharmaceuticals", "Healthcare", "Generics"],
    "maruti": ["Automotive", "Manufacturing", "Finance"],
    "hero": ["Automotive", "Manufacturing", "Finance"],
    "bharti airtel": ["Telecom", "Media", "Finance", "Infrastructure"],
    "airtel": ["Telecom", "Media", "Finance"],
    "jio": ["Telecom", "Retail", "Media", "Finance"],
    "zomato": ["Food Delivery", "Technology", "Logistics"],
    "swiggy": ["Food Delivery", "Technology", "Logistics"],
    "paytm": ["Fintech", "Payments", "Banking", "Insurance"],
    "ola": ["Mobility", "Technology", "Finance", "EV"],
    "flipkart": ["E-commerce", "Logistics", "Payments", "Technology"],
    "amazon": ["E-commerce", "Cloud", "Logistics", "Media", "AI"],
    "google": ["Technology", "Advertising", "Cloud", "AI", "Hardware"],
    "apple": ["Technology", "Hardware", "Software", "Services", "AI"],
    "microsoft": ["Technology", "Cloud", "AI", "Gaming", "Enterprise"],
    "meta": ["Social Media", "Advertising", "AI", "VR", "Technology"],
    "tesla": ["Automotive", "Energy", "AI", "Technology"],
    "nvidia": ["Semiconductors", "AI", "Gaming", "Data Centers"],
    "samsung": ["Electronics", "Semiconductors", "Appliances", "Display"],
    "toyota": ["Automotive", "Finance", "Robotics", "Energy"],
    "jp morgan": ["Banking", "Finance", "Investment", "Asset Management"],
    "goldman sachs": ["Banking", "Investment", "Finance", "Asset Management"],
}

DEFAULT_SECTORS = ["Business", "Finance", "Technology", "Operations"]


def get_company_sectors(company_name: str) -> list[str]:
    """
    Returns a list of business sectors for a given company name.
    Falls back to default sectors if the company is not in the lookup.
    """
    key = company_name.lower().strip()

    if key in COMPANY_SECTORS:
        return COMPANY_SECTORS[key]

    for company, sectors in COMPANY_SECTORS.items():
        if company in key or key in company:
            return sectors

    return DEFAULT_SECTORS