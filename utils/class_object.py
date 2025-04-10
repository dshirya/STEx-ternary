import re
import pandas as pd

class SiteElement:
    """
    Represents a material's site information using hardcoded DataFrame columns.

    This class assumes the DataFrame row has the following columns:
      - "Filename"
      - "Formula"
      - "2c"
      - "6h"
      - "RE"
    
    The site columns ("2c", "6h", "RE") are parsed so that if a cell contains
    a mixture of elements (e.g., "Cd0.114Tm2.886"), the element with the larger numeric
    count is selected as the primary element.
    
    The formula is also parsed and the primary element (by highest count) is stored in
    self.elements.
    """
    
    def __init__(self, row):
        self.filename = row["Filename"]
        self.formula = row["Formula"]
        
        self.site_2c = self.get_primary_element(row["2c"])
        self.site_6h2 = self.get_primary_element(row["6h"])
        self.site_RE = self.get_primary_element(row["RE"])
        
        # Parse the formula and save its primary element in self.elements
        self.elements = self.parse_formula(self.formula)
    
    @staticmethod
    def parse_elements(cell_value):
        """
        Parses a string that contains one or more element symbols along with their numeric counts.
        Uses the regex pattern: r'([A-Z][a-z]*)(\d*\.?\d*)'
        
        Examples:
          - "Tm" returns [('Tm', 1.0)]
          - "Tm0.886Cd0.114" returns [('Tm', 0.886), ('Cd', 0.114)]
        
        If the numeric part is missing, defaults to a count of 1.0.
        
        Parameters:
            cell_value (str): The cell content from one of the element columns.
            
        Returns:
            list of tuples: A list containing (element, count) pairs.
        """
        s = str(cell_value).strip()
        if not s:
            return []
        pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
        matches = re.findall(pattern, s)
        results = []
        for element, count_str in matches:
            count = float(count_str) if count_str != "" else 1.0
            if element:  # Avoid empty matches
                results.append((element, count))
        return results

    def get_primary_element(self, cell_value):
        """
        Parses a site cell value and returns its primary element.
        
        In cases of a mixture (e.g., "Cd0.114Tm2.886"), the element with the larger
        count is returned.
        
        Parameters:
            cell_value (str): The string containing element symbols and their counts.
            
        Returns:
            str: The primary element symbol (or None if not found).
        """
        parsed_items = self.parse_elements(cell_value)
        if not parsed_items:
            return None
        # Select the element with the highest numeric count
        primary_element, _ = max(parsed_items, key=lambda x: x[1])
        return primary_element
    
    def parse_formula(self, formula):
        """
        Parses the compound formula and returns its primary element.
        
        If multiple elements are present, the element with the highest numeric count is chosen.
        
        Parameters:
            formula (str): The compound formula string.
            
        Returns:
            str: The primary element symbol from the formula (or None if not found).
        """
        parsed_items = self.parse_elements(formula)
        if not parsed_items:
            return None
        primary_element, _ = max(parsed_items, key=lambda x: x[1])
        return primary_element
