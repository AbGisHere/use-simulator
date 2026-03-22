"""
Sector-level news taxonomy for NSE tickers.

Maps each ticker to three keyword buckets:
  - company_terms : direct company name queries (highest weight)
  - sector_terms  : industry / sector queries (medium weight)
  - proxy_terms   : proxy events / government schemes / commodities (lower weight)

News fetched using company_terms gets match_type='company'  (weight 1.0)
News fetched using sector_terms  gets match_type='sector'   (weight 0.6)
News fetched using proxy_terms   gets match_type='proxy'    (weight 0.4)

These weights are applied in features/sentiment.py during FinBERT aggregation.
"""

from typing import TypedDict


class TickerTaxonomy(TypedDict):
    company_terms: list[str]   # Full text queries for Google News RSS
    sector_terms: list[str]    # Sector/industry queries
    proxy_terms: list[str]     # Commodity/policy/event proxy queries


# ---------------------------------------------------------------------------
# Taxonomy dictionary  (symbol → TickerTaxonomy)
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, TickerTaxonomy] = {

    # ── IT / Software ──────────────────────────────────────────────────────
    "TCS": {
        "company_terms": ["Tata Consultancy Services TCS NSE"],
        "sector_terms": ["India IT sector Infosys Wipro HCL earnings", "India software exports"],
        "proxy_terms": ["US dollar rupee exchange rate India", "H1B visa India IT", "US tech layoffs India"],
    },
    "INFY": {
        "company_terms": ["Infosys NSE India"],
        "sector_terms": ["India IT sector TCS Wipro HCL results", "India software exports"],
        "proxy_terms": ["US dollar rupee exchange rate India", "H1B visa India IT", "US recession India IT"],
    },
    "WIPRO": {
        "company_terms": ["Wipro NSE India"],
        "sector_terms": ["India IT sector TCS Infosys HCL results", "India software exports"],
        "proxy_terms": ["US dollar rupee exchange rate India", "H1B visa India IT"],
    },
    "HCLTECH": {
        "company_terms": ["HCL Technologies NSE India"],
        "sector_terms": ["India IT sector TCS Infosys results", "India software exports"],
        "proxy_terms": ["US dollar rupee exchange rate India", "H1B visa India IT"],
    },
    "TECHM": {
        "company_terms": ["Tech Mahindra NSE India"],
        "sector_terms": ["India IT sector results", "India telecom IT"],
        "proxy_terms": ["US dollar rupee India", "5G India technology"],
    },
    "MPHASIS": {
        "company_terms": ["Mphasis NSE India"],
        "sector_terms": ["India mid-cap IT results", "India software exports"],
        "proxy_terms": ["US dollar rupee India", "US financial services IT"],
    },

    # ── Banking & Finance ──────────────────────────────────────────────────
    "HDFCBANK": {
        "company_terms": ["HDFC Bank NSE India"],
        "sector_terms": ["India banking sector RBI credit growth", "Indian bank NPA"],
        "proxy_terms": ["RBI repo rate India", "India inflation CPI RBI policy", "India GDP growth banking"],
    },
    "ICICIBANK": {
        "company_terms": ["ICICI Bank NSE India"],
        "sector_terms": ["India banking sector RBI credit growth", "Indian bank NPA"],
        "proxy_terms": ["RBI repo rate India", "India inflation CPI RBI policy"],
    },
    "SBIN": {
        "company_terms": ["State Bank of India SBI NSE"],
        "sector_terms": ["India PSU banking sector RBI credit", "Indian public sector bank"],
        "proxy_terms": ["RBI repo rate India", "India government spending bank", "India infrastructure loan"],
    },
    "AXISBANK": {
        "company_terms": ["Axis Bank NSE India"],
        "sector_terms": ["India private banking sector RBI credit growth"],
        "proxy_terms": ["RBI repo rate India", "India retail loan credit card"],
    },
    "KOTAKBANK": {
        "company_terms": ["Kotak Mahindra Bank NSE India"],
        "sector_terms": ["India private banking sector RBI results"],
        "proxy_terms": ["RBI repo rate India", "India wealth management banking"],
    },
    "BAJFINANCE": {
        "company_terms": ["Bajaj Finance NSE India"],
        "sector_terms": ["India NBFC sector loan growth credit", "India consumer finance"],
        "proxy_terms": ["RBI NBFC regulation India", "India retail consumer spending loan"],
    },
    "BAJAJFINSV": {
        "company_terms": ["Bajaj Finserv NSE India"],
        "sector_terms": ["India insurance NBFC sector", "India financial services"],
        "proxy_terms": ["RBI regulation India insurance", "India health insurance penetration"],
    },
    "SBICARD": {
        "company_terms": ["SBI Cards NSE India"],
        "sector_terms": ["India credit card market growth", "India digital payments"],
        "proxy_terms": ["RBI repo rate hike India credit card", "India UPI credit card EMI"],
    },
    "HDFCLIFE": {
        "company_terms": ["HDFC Life Insurance NSE India"],
        "sector_terms": ["India life insurance sector premium growth", "IRDAI India"],
        "proxy_terms": ["India budget tax insurance", "India insurance penetration policy"],
    },
    "SBILIFE": {
        "company_terms": ["SBI Life Insurance NSE India"],
        "sector_terms": ["India life insurance sector premium growth", "IRDAI India"],
        "proxy_terms": ["India budget tax insurance", "India mortality rate insurance"],
    },

    # ── Oil, Gas & Energy ──────────────────────────────────────────────────
    "RELIANCE": {
        "company_terms": ["Reliance Industries NSE India"],
        "sector_terms": ["India oil gas sector refinery", "India petrochemicals Jio retail"],
        "proxy_terms": ["crude oil price Brent WTI India", "India natural gas price LNG", "India 5G telecom Jio spectrum"],
    },
    "ONGC": {
        "company_terms": ["ONGC NSE India Oil Natural Gas"],
        "sector_terms": ["India upstream oil gas sector exploration", "India PSU energy"],
        "proxy_terms": ["crude oil price Brent India", "India gas field discovery", "OPEC production cut India"],
    },
    "BPCL": {
        "company_terms": ["Bharat Petroleum BPCL NSE India"],
        "sector_terms": ["India OMC oil marketing company fuel prices", "India petroleum refinery"],
        "proxy_terms": ["crude oil price India", "India petrol diesel price revision", "India LPG cylinder price"],
    },
    "IOC": {
        "company_terms": ["Indian Oil IOC NSE India"],
        "sector_terms": ["India OMC oil marketing fuel prices refinery", "India PSU petroleum"],
        "proxy_terms": ["crude oil price India", "India petrol diesel price", "India LPG subsidy"],
    },
    "HINDPETRO": {
        "company_terms": ["Hindustan Petroleum HPCL NSE India"],
        "sector_terms": ["India OMC oil marketing company fuel prices"],
        "proxy_terms": ["crude oil price India", "India petrol diesel price revision"],
    },
    "NTPC": {
        "company_terms": ["NTPC NSE India power electricity"],
        "sector_terms": ["India power electricity sector coal capacity", "India PSU power generation"],
        "proxy_terms": ["India coal price shortage power", "India renewable energy solar target", "India electricity demand summer"],
    },
    "POWERGRID": {
        "company_terms": ["Power Grid India NSE"],
        "sector_terms": ["India power transmission sector", "India electricity grid infrastructure"],
        "proxy_terms": ["India renewable energy solar wind transmission", "India power capex budget"],
    },
    "TATAPOWER": {
        "company_terms": ["Tata Power NSE India"],
        "sector_terms": ["India power generation renewable solar", "India electricity sector"],
        "proxy_terms": ["India solar energy policy PLI", "India EV charging infrastructure", "coal price India"],
    },
    "ADANIGREEN": {
        "company_terms": ["Adani Green Energy NSE India"],
        "sector_terms": ["India renewable energy solar wind capacity", "India green energy sector"],
        "proxy_terms": ["India solar power policy PLI scheme", "India renewable energy target 2030"],
    },
    "ADANIPORTS": {
        "company_terms": ["Adani Ports SEZ NSE India"],
        "sector_terms": ["India port logistics infrastructure sector", "India shipping cargo"],
        "proxy_terms": ["India export import trade volume", "India maritime policy Sagarmala", "Red Sea shipping India"],
    },
    "ADANIENT": {
        "company_terms": ["Adani Enterprises NSE India"],
        "sector_terms": ["India conglomerate infrastructure energy sector", "Adani Group India"],
        "proxy_terms": ["India airport privatisation", "India coal import price", "India infrastructure budget capex"],
    },

    # ── Auto ───────────────────────────────────────────────────────────────
    "MARUTI": {
        "company_terms": ["Maruti Suzuki NSE India"],
        "sector_terms": ["India auto sector car sales passenger vehicle", "SIAM India auto"],
        "proxy_terms": ["India petrol diesel price car demand", "India festive season car sales", "India EV policy FAME subsidy", "steel price India auto"],
    },
    "TATAMOTORS": {
        "company_terms": ["Tata Motors NSE India"],
        "sector_terms": ["India auto sector EV commercial vehicle", "India PV CV sales SIAM"],
        "proxy_terms": ["India EV policy FAME scheme", "steel aluminium price India auto", "India festive season vehicle sales"],
    },
    "M&M": {
        "company_terms": ["Mahindra Mahindra NSE India"],
        "sector_terms": ["India auto SUV tractor sector", "India farm equipment tractor sales"],
        "proxy_terms": ["India monsoon crop farm income tractor", "India SUV demand EV", "steel price India auto"],
    },
    "HEROMOTOCO": {
        "company_terms": ["Hero MotoCorp NSE India"],
        "sector_terms": ["India two-wheeler sector motorcycle scooter sales"],
        "proxy_terms": ["India rural income farm sector motorcycle", "India petrol price two-wheeler", "India festive season two-wheeler sales"],
    },
    "BAJAJ-AUTO": {
        "company_terms": ["Bajaj Auto NSE India"],
        "sector_terms": ["India two-wheeler three-wheeler sector exports"],
        "proxy_terms": ["India motorcycle exports Africa", "India petrol price two-wheeler", "India rural demand"],
    },
    "EICHERMOT": {
        "company_terms": ["Eicher Motors Royal Enfield NSE India"],
        "sector_terms": ["India premium motorcycle segment", "India two-wheeler sector"],
        "proxy_terms": ["India rural income premium bike demand", "India highway infrastructure tourism"],
    },

    # ── Metals & Mining ────────────────────────────────────────────────────
    "TATASTEEL": {
        "company_terms": ["Tata Steel NSE India"],
        "sector_terms": ["India steel sector prices capacity", "India metal mining sector"],
        "proxy_terms": ["China steel production export price India", "India infrastructure government capex steel", "iron ore price India"],
    },
    "JSWSTEEL": {
        "company_terms": ["JSW Steel NSE India"],
        "sector_terms": ["India steel sector prices capacity production", "India metal sector"],
        "proxy_terms": ["China steel export dump India", "India infrastructure capex steel demand", "coking coal price India"],
    },
    "HINDALCO": {
        "company_terms": ["Hindalco Industries NSE India"],
        "sector_terms": ["India aluminium copper sector", "India non-ferrous metal"],
        "proxy_terms": ["aluminium LME price India", "copper price India", "India EV aluminium demand"],
    },
    "VEDL": {
        "company_terms": ["Vedanta NSE India"],
        "sector_terms": ["India zinc copper aluminium mining sector"],
        "proxy_terms": ["zinc LME price India", "India mining royalty regulation", "crude oil Cairn India"],
    },
    "COALINDIA": {
        "company_terms": ["Coal India NSE"],
        "sector_terms": ["India coal production power sector", "India thermal coal"],
        "proxy_terms": ["India power sector coal demand shortage", "India renewable energy coal displacement", "Indonesia coal price India"],
    },

    # ── FMCG / Consumer ────────────────────────────────────────────────────
    "HINDUNILVR": {
        "company_terms": ["Hindustan Unilever HUL NSE India"],
        "sector_terms": ["India FMCG sector rural urban consumption", "India consumer staples"],
        "proxy_terms": ["India monsoon rural demand FMCG", "India crude palm oil price FMCG", "India GST rate FMCG"],
    },
    "NESTLEIND": {
        "company_terms": ["Nestle India NSE"],
        "sector_terms": ["India packaged food FMCG sector", "India branded food consumption"],
        "proxy_terms": ["India cocoa sugar price food", "India GST packaged food", "India urban consumption"],
    },
    "DABUR": {
        "company_terms": ["Dabur India NSE"],
        "sector_terms": ["India Ayurveda health FMCG sector", "India herbal consumer goods"],
        "proxy_terms": ["India monsoon rural demand Ayurveda", "India healthcare wellness trend"],
    },
    "MARICO": {
        "company_terms": ["Marico NSE India"],
        "sector_terms": ["India FMCG hair oil edible oil sector"],
        "proxy_terms": ["India copra coconut price oil", "India palm oil price edible oil"],
    },
    "GODREJCP": {
        "company_terms": ["Godrej Consumer Products NSE India"],
        "sector_terms": ["India FMCG household insecticides sector"],
        "proxy_terms": ["India monsoon vector-borne disease insecticide demand", "India urban rural FMCG"],
    },
    "ITC": {
        "company_terms": ["ITC NSE India"],
        "sector_terms": ["India cigarettes hotels FMCG sector", "India tobacco policy"],
        "proxy_terms": ["India cigarette excise duty budget", "India hotel tourism recovery", "India wheat agri price ITC"],
    },

    # ── Pharma / Healthcare ────────────────────────────────────────────────
    "SUNPHARMA": {
        "company_terms": ["Sun Pharmaceutical NSE India"],
        "sector_terms": ["India pharma sector generic drug exports US FDA"],
        "proxy_terms": ["US FDA inspection India pharma", "India drug price control NPPA", "rupee dollar India pharma exports"],
    },
    "DRREDDY": {
        "company_terms": ["Dr Reddys Laboratories NSE India"],
        "sector_terms": ["India pharma exports US generics FDA India"],
        "proxy_terms": ["US FDA warning letter India pharma", "India drug price control NPPA", "rupee dollar pharma"],
    },
    "CIPLA": {
        "company_terms": ["Cipla NSE India pharmaceuticals"],
        "sector_terms": ["India pharma respiratory HIV generics sector"],
        "proxy_terms": ["US FDA inspection India pharma", "India drug price NPPA", "India healthcare policy"],
    },
    "DIVISLAB": {
        "company_terms": ["Divi's Laboratories NSE India"],
        "sector_terms": ["India API active pharmaceutical ingredient sector", "India pharma contract manufacturing"],
        "proxy_terms": ["China API supply India pharma", "India pharma CDMO sector policy"],
    },
    "APOLLOHOSP": {
        "company_terms": ["Apollo Hospitals NSE India"],
        "sector_terms": ["India private hospital healthcare sector", "India medical tourism"],
        "proxy_terms": ["India health insurance Ayushman Bharat", "India doctor shortage infrastructure healthcare"],
    },

    # ── Telecom ────────────────────────────────────────────────────────────
    "BHARTIARTL": {
        "company_terms": ["Bharti Airtel NSE India"],
        "sector_terms": ["India telecom sector ARPU 5G spectrum", "India mobile broadband"],
        "proxy_terms": ["India 5G rollout spectrum auction TRAI", "India telecom tariff hike", "India broadband fibre"],
    },

    # ── Cement ────────────────────────────────────────────────────────────
    "ULTRACEMCO": {
        "company_terms": ["UltraTech Cement NSE India"],
        "sector_terms": ["India cement sector demand production prices"],
        "proxy_terms": ["India infrastructure government spending housing cement", "coal price India cement", "India real estate housing demand"],
    },
    "AMBUJACEM": {
        "company_terms": ["Ambuja Cements NSE India"],
        "sector_terms": ["India cement sector demand production"],
        "proxy_terms": ["India housing infrastructure cement demand", "coal price India cement cost"],
    },
    "ACC": {
        "company_terms": ["ACC Limited NSE India cement"],
        "sector_terms": ["India cement sector demand production"],
        "proxy_terms": ["India housing infrastructure cement demand", "coal price India cement"],
    },

    # ── Construction / Infrastructure ─────────────────────────────────────
    "LT": {
        "company_terms": ["Larsen Toubro L&T NSE India"],
        "sector_terms": ["India infrastructure EPC sector order book", "India government capex construction"],
        "proxy_terms": ["India budget infrastructure allocation", "India defence order", "India metro rail project", "India water project Jal Jeevan"],
    },
    "DLF": {
        "company_terms": ["DLF NSE India real estate"],
        "sector_terms": ["India real estate residential commercial sector", "India property market"],
        "proxy_terms": ["India repo rate home loan housing demand", "India stamp duty real estate policy", "India luxury housing demand"],
    },

    # ── Entertainment / Media ──────────────────────────────────────────────
    "PVR": {
        "company_terms": ["PVR INOX NSE India multiplex"],
        "sector_terms": ["India multiplex cinema box office Bollywood", "India theatre chain"],
        "proxy_terms": ["Bollywood Hindi film box office collection India", "India OTT streaming vs theatrical window", "Hollywood release India box office"],
    },
    "INOXLEISUR": {
        "company_terms": ["INOX Leisure NSE India multiplex"],
        "sector_terms": ["India multiplex cinema box office Bollywood sector"],
        "proxy_terms": ["Bollywood Hindi film box office India", "India OTT direct release vs cinema", "Hollywood India box office"],
    },
    "ZEEL": {
        "company_terms": ["Zee Entertainment NSE India"],
        "sector_terms": ["India media television OTT sector", "India broadcast advertising"],
        "proxy_terms": ["India digital advertising market", "India OTT subscriber growth", "India television viewership"],
    },

    # ── Aviation ──────────────────────────────────────────────────────────
    "INDIGO": {
        "company_terms": ["IndiGo InterGlobe Aviation NSE India"],
        "sector_terms": ["India aviation airline sector passenger load factor", "India domestic air travel"],
        "proxy_terms": ["India jet fuel ATF price", "India aviation demand summer winter", "India airport infrastructure expansion"],
    },
    "SPICEJET": {
        "company_terms": ["SpiceJet NSE India airline"],
        "sector_terms": ["India aviation airline sector low cost carrier", "India air travel"],
        "proxy_terms": ["India jet fuel ATF price", "India aviation DGCA regulation", "India domestic travel demand"],
    },

    # ── Hotels / Hospitality ───────────────────────────────────────────────
    "INDHOTEL": {
        "company_terms": ["Indian Hotels Taj Hotels NSE India"],
        "sector_terms": ["India hospitality hotel sector occupancy RevPAR", "India tourism"],
        "proxy_terms": ["India domestic foreign tourist arrivals", "India G20 MICE tourism", "India wedding season hotel demand"],
    },
    "LEMONTREE": {
        "company_terms": ["Lemon Tree Hotels NSE India"],
        "sector_terms": ["India budget hotel hospitality sector"],
        "proxy_terms": ["India domestic tourism travel demand", "India staycation hotel mid-market"],
    },

    # ── Chemicals ──────────────────────────────────────────────────────────
    "PIDILITIND": {
        "company_terms": ["Pidilite Industries Fevicol NSE India"],
        "sector_terms": ["India adhesive sealant specialty chemical sector", "India construction chemicals"],
        "proxy_terms": ["India real estate housing construction activity Pidilite", "vinyl acetate monomer price India"],
    },
    "AARTIIND": {
        "company_terms": ["Aarti Industries NSE India"],
        "sector_terms": ["India specialty chemicals benzene toluene sector"],
        "proxy_terms": ["China chemical supply India specialty", "benzene price India chemicals"],
    },
    "SRF": {
        "company_terms": ["SRF Limited NSE India"],
        "sector_terms": ["India specialty chemical fluorochemical refrigerant sector"],
        "proxy_terms": ["India refrigerant HFC policy regulation", "India technical textile exports"],
    },
    "RAIN": {
        "company_terms": ["Rain Industries NSE India"],
        "sector_terms": ["India carbon black petroleum refinery sector"],
        "proxy_terms": ["coal tar pitch price India", "India aluminium smelting carbon anode", "crude oil refinery by-product India"],
    },

    # ── E-commerce / Internet ──────────────────────────────────────────────
    "ZOMATO": {
        "company_terms": ["Zomato NSE India food delivery"],
        "sector_terms": ["India food delivery quick commerce sector Swiggy Blinkit", "India internet consumer"],
        "proxy_terms": ["India fuel price delivery cost logistics", "India urban consumption food order", "India gig worker regulation"],
    },
    "NYKAA": {
        "company_terms": ["Nykaa FSN E-Commerce NSE India"],
        "sector_terms": ["India beauty cosmetics e-commerce sector", "India D2C brand"],
        "proxy_terms": ["India urban women spending beauty cosmetics", "India festive sale e-commerce"],
    },

    # ── Diversified / Conglomerate ─────────────────────────────────────────
    "TITAN": {
        "company_terms": ["Titan Company NSE India jewellery watches"],
        "sector_terms": ["India jewellery watches retail sector", "India gold demand consumer"],
        "proxy_terms": ["India gold price import duty jewellery", "India wedding festive season jewellery demand", "India luxury consumer spending"],
    },

    # ── Defence ────────────────────────────────────────────────────────────
    "HAL": {
        "company_terms": ["Hindustan Aeronautics HAL NSE India"],
        "sector_terms": ["India defence aerospace PSU sector", "India military procurement"],
        "proxy_terms": ["India defence budget capex", "India Tejas fighter aircraft export", "India Make in India defence policy"],
    },
    "BEL": {
        "company_terms": ["Bharat Electronics BEL NSE India"],
        "sector_terms": ["India defence electronics PSU sector"],
        "proxy_terms": ["India defence budget modernisation", "India Make in India defence order"],
    },

    # ── Agri / Fertiliser ──────────────────────────────────────────────────
    "COROMANDEL": {
        "company_terms": ["Coromandel International NSE India fertiliser"],
        "sector_terms": ["India fertiliser agrochemical sector crop protection"],
        "proxy_terms": ["India monsoon crop sowing kharif rabi fertiliser", "India fertiliser subsidy budget", "India urea DAP price"],
    },
    "CHAMBLFERT": {
        "company_terms": ["Chambal Fertilisers NSE India"],
        "sector_terms": ["India fertiliser urea sector production"],
        "proxy_terms": ["India urea price subsidy", "India natural gas price fertiliser production", "India monsoon farm demand"],
    },
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_taxonomy(ticker: str) -> TickerTaxonomy | None:
    """Return taxonomy for a ticker (strips .NS / .BO suffix)."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    return TAXONOMY.get(symbol)


def get_all_query_terms(ticker: str) -> dict[str, list[str]]:
    """
    Return all search terms for a ticker, keyed by match_type.
    Falls back to a generic query if ticker not in taxonomy.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    taxonomy = TAXONOMY.get(symbol)

    if taxonomy:
        return {
            "company": taxonomy["company_terms"],
            "sector": taxonomy["sector_terms"],
            "proxy": taxonomy["proxy_terms"],
        }

    # Generic fallback for unlisted tickers
    return {
        "company": [f"{symbol} NSE India stock"],
        "sector": [],
        "proxy": [],
    }
