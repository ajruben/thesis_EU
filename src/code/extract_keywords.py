#imports
import argparse
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging #TODO, setup loggin? not set up yet
#suggested libs:
#use this for warnings, so we know when conditions are not met
import warnings
#use this for parsing urls
from urllib.parse import urlparse, urljoin
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from pathlib import Path

#setup class
class KeywordExtractor():	
    def __init__(self, keyword_attr : str = "keywords", tag : str = "meta"):
        self.BASE_CORDIS_URL = "https://cordis.europa.eu/project/id/"
        self.keyword_attr = keyword_attr #attr page with keyword
        self.tag = tag                   #tag with keywords
        
    def scrape_url(self, 
                    project_id: str,
                    base_url :str = None,
                    retries: int = 3,
                    backoff: float = 2,
                    timeout: float = 10) -> list[str] | str:
        """
        TODO
        """ 
        #url of cordis project page
        if base_url is None:
            url = urljoin(self.BASE_CORDIS_URL, str(project_id))
        else:
            url = urljoin(base_url, project_id)
        
        # source original setup: https://oxylabs.io/blog/python-requests-retry 
        retry_setup = Retry(
            total=retries,
            backoff_factor=backoff, #{backoff factor} * (2 ** ({number of previous retries}))
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_setup)
        session = requests.Session()
        session.mount('https://', adapter)
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()  
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Request failed: {e}", Warning)
            return "ERROR"
        
        cordis_html_txt = r.text
        self.project_id = project_id
        return cordis_html_txt
    
    def parse_keywords_from_html(self, 
                                 html_txt: str,
                                 attrs: str = "keywords") -> list[str] | str:
        """
        -explain what cordis is
        -explain what the function does
        -explain variables
        -explain return 
        """
        #parse html with BeautifulSoup
        soup = BeautifulSoup(html_txt, "html.parser")
        
        #find tag with keywords, <meta name="keywords" .... 
        keywords_soup = soup.find(self.tag, attrs={"name": self.keyword_attr})
        if keywords_soup is None:
            warnings.warn(f"Could not find keywords for project {self.project_id}.", Warning)
            return "NOT FOUND"

        keywords = set([kw.strip() for kw in keywords_soup["content"].split(",") if (kw != "" and not kw.startswith("HORIZON"))])
        return list(keywords)

class KeywordExtractorEU(KeywordExtractor):
    def __init__(self, keyword_attr = "keywords", tag = "meta", cli=True):
        super().__init__(keyword_attr, tag)
        #setup proj dirs
        self.ROOT = Path(__file__).resolve(strict=True).parent.parent
        self.DATA_DIR = self.ROOT / "datasets"
        self.OUT_DIR = self.ROOT / "out"
        #def csv file locations
        self.default_project_file = self.DATA_DIR / 'project.csv'
        self.default_euroscivoc_file = self.DATA_DIR / 'euroscivoc.csv'
        # parse command line arguments if cli, setup clusters
        if cli:
            self._setup_cli_parser()
            self.clusters = self._cliparse_clusters()    
        else:
            #TODO
            self._setup_args()
        
    def process_csv_files(self) -> pd.DataFrame:
        """
        -explain what euroscivoc is
        -explain what the function does
        """

        #read project file, drop columns not needed, convert numbers to floats and filter clusters
        project_df = pd.read_csv(self.args.projectfile)
        project_df = project_df.drop(columns=[
            "acronym", "status", "title", "startDate", "endDate", "totalCost",
            "legalBasis", "ecSignatureDate", "frameworkProgramme",
            "masterCall", "subCall", "fundingScheme", "nature", "objective",
            "contentUpdateDate", "rcn", "grantDoi"
        ])
        project_df["ecMaxContribution"] = project_df["ecMaxContribution"].apply(lambda val: self._convert_to_float(val))
        project_df["cluster"] = project_df["topics"].apply(lambda topic: self._get_cluster(topic))
        if self.clusters != "all":
            project_df = project_df.loc[project_df.cluster.isin(self.clusters)]

        # process euro scientific vocab file, add keywords to proj df
        print("Getting euroSciVoc keywords...")
        euroscivoc_df = pd.read_csv(self.args.euroscivocfile)
        euroscivoc_keywords = euroscivoc_df.groupby("projectID")["euroSciVocTitle"].apply(list).reset_index(name="euroscivoc_keywords")
        project_df = project_df.merge(euroscivoc_keywords, how="left", left_on="id", right_on="projectID").drop(columns=["projectID"])
        return project_df

    def get_cordis_keywords(self, project_df: pd.DataFrame, save = True) -> list[str] | str:
        tqdm.pandas(desc="Retrieving keywords from CORDIS", leave=True, miniters=1)
        project_df["cordis_keywords"] = project_df.progress_apply(lambda row: self._get_cordis_keywords_scrape(row.id), axis=1)
        
        if save:
            out_file = "./out/extracted.csv"
            print(f"Writing to {out_file} ...")
            try:
                os.makedirs("./out", exist_ok=True)
                project_df.to_csv(out_file, index=False)
            except FileExistsError:
                pass
            
        return project_df    
    
    def _get_cordis_keywords_scrape(self, project_id: str) -> list[str] | str:
        html_txt = self.scrape_url(project_id)
        if html_txt == "ERROR":
            return "ERROR"
        else:
            keywords = self.parse_keywords_from_html(html_txt)
            return keywords

    #-- helper functions
    def _cliparse_clusters(self) -> list[str] | str:
        # parse clusters
        if self.args.clusters == "all":
            clusters = "all"
        else:
            clusters = []
            for c in self.args.clusters:
                try:
                    c = int(c)
                    if c < 1 or c > 6:
                        raise ValueError("--clusters/-c can only contain numbers 1-6")
                    clusters.append(f"CL{c}")
                except ValueError as ve:
                    self.cli_parser.error(f"parserse error: {ve}")
        return clusters
    
    def _convert_to_float(self, val: str | int) -> float | str | None:
        if pd.isna(val):
            return None
        try:
            return float(str(val).replace(",", "."))
        except ValueError as ve:
            warnings.warn(f"Could not convert {val} to float: {ve}, returning {val}.", Warning)
            return val

    def _get_cluster(self, topic: str) -> str:
        if topic.startswith("HORIZON"):
            return "HORIZON-" + topic.split("-")[1]
        elif topic.startswith("ERC"):
            return "ERC-" + topic.split("-")[2]
        elif topic.startswith("EURATOM"):
            return "EURATOM-" + "-".join(topic.split("-")[2:-1])
        else:
            warnings.warn(f"Can't find topic format: {topic}. returning as is.", Warning)
            return topic

    def _get_euroscivoc_keywords(self, 
                                project_id: str, 
                                euroscivoc_df: pd.DataFrame) -> list[str]:
        """
        -explain what euroscivoc is
        -explain what the function does
        """
        euroscivoc_keywords  = euroscivoc_df.loc[euroscivoc_df["projectID"] == project_id]["euroSciVocTitle"].tolist()
        #maybe some checks? otherwise method not needed.
        return euroscivoc_keywords 
    
    def _setup_cli_parser(self):
        self.cli_parser = argparse.ArgumentParser()
        self.cli_parser.add_argument("--clusters", "-c", nargs="?", const="all", default="all", type=str, help="clusters to look at (e.g. 124 for clusters 1, 2 and 4)")
        self.cli_parser.add_argument("--projectfile", "-pf", nargs='?', const=self.default_project_file, default=self.default_project_file, type=str)
        self.cli_parser.add_argument("--euroscivocfile", "-ef", nargs="?", const=self.default_euroscivoc_file, default=self.default_euroscivoc_file, type=str)
        self.args = self.cli_parser.parse_args()
    
    def _setup_args(self):
        self.args = argparse.Namespace()
        self.args.clusters = "all"
        self.args.projectfile = self.default_project_file
        self.args.euroscivocfile = self.default_euroscivoc_file
    
    #-- main function
    def run(self):
        project_df = self.process_csv_files()
        project_df = self.get_cordis_keywords(project_df)
        

if __name__ == "__main__":
    keyword_scraper = KeywordExtractorEU()
    keyword_scraper.run()
    

    