"""The feature contract shared by the data gate, the model file and the API.

One definition, three enforcement points: the training data must declare exactly these
attributes and domains, an exported model must list exactly these features in this order,
and the API rejects any request that does not match them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    values: tuple[int, ...]
    description: str


# Names are kept verbatim from the dataset (typos included) so every artefact can be traced
# back to the source file. Values use the authors' ternary encoding.
FEATURES: tuple[Feature, ...] = (
    Feature("having_IP_Address", (-1, 1), "IP address used instead of a domain name"),
    Feature("URL_Length", (-1, 0, 1), "length of the URL"),
    Feature("Shortining_Service", (-1, 1), "URL shortening service"),
    Feature("having_At_Symbol", (-1, 1), "'@' symbol in the URL"),
    Feature("double_slash_redirecting", (-1, 1), "'//' inside the URL path"),
    Feature("Prefix_Suffix", (-1, 1), "prefix or suffix separated by '-' in the domain"),
    Feature("having_Sub_Domain", (-1, 0, 1), "number of sub-domains"),
    Feature("SSLfinal_State", (-1, 0, 1), "HTTPS, certificate issuer and certificate age"),
    Feature("Domain_registeration_length", (-1, 1), "domain registration length"),
    Feature("Favicon", (-1, 1), "favicon loaded from another domain"),
    Feature("port", (-1, 1), "non-standard ports open"),
    Feature("HTTPS_token", (-1, 1), "'https' token in the domain part"),
    Feature("Request_URL", (-1, 1), "share of embedded objects loaded from other domains"),
    Feature("URL_of_Anchor", (-1, 0, 1), "share of anchors pointing elsewhere or nowhere"),
    Feature("Links_in_tags", (-1, 0, 1), "share of <meta>, <script>, <link> links elsewhere"),
    Feature("SFH", (-1, 0, 1), "server form handler empty, blank or on another domain"),
    Feature("Submitting_to_email", (-1, 1), "form submitted to an e-mail address"),
    Feature("Abnormal_URL", (-1, 1), "host name absent from the URL"),
    Feature("Redirect", (0, 1), "number of redirects"),
    Feature("on_mouseover", (-1, 1), "onMouseOver rewrites the status bar"),
    Feature("RightClick", (-1, 1), "right click disabled"),
    Feature("popUpWidnow", (-1, 1), "pop-up window with text fields"),
    Feature("Iframe", (-1, 1), "invisible iframe"),
    Feature("age_of_domain", (-1, 1), "age of the domain"),
    Feature("DNSRecord", (-1, 1), "DNS record for the domain"),
    Feature("web_traffic", (-1, 0, 1), "website traffic rank"),
    Feature("Page_Rank", (-1, 1), "PageRank"),
    Feature("Google_Index", (-1, 1), "page indexed by Google"),
    Feature("Links_pointing_to_page", (-1, 0, 1), "number of links pointing to the page"),
    Feature("Statistical_report", (-1, 1), "host listed in phishing statistics reports"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
N_FEATURES = len(FEATURES)

LABEL = "Result"
LABEL_VALUES: tuple[int, ...] = (-1, 1)
PHISHING = -1  # dataset encoding of the label: -1 = phishing, 1 = legitimate


def declared_attributes() -> tuple[tuple[str, tuple[int, ...]], ...]:
    """The attribute list a conforming data file must declare, label last."""
    return (*((f.name, f.values) for f in FEATURES), (LABEL, LABEL_VALUES))
