import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

def _parse_formula(formula: str, strict: bool = True) -> dict:
    """
    Parse a chemical formula (e.g. 'Fe2O3', 'Li3Fe2(PO4)3') into its constituent elements and their counts.
    """
    if "'" in formula:
        formula = formula.replace("'", "")
    if strict and re.match(r"[\s\d.*/]*$", formula):
        raise ValueError(f"Invalid formula: {formula}")
    # Handle metallofullerenes and coordination complexes
    formula = formula.replace("@", "")
    formula = formula.replace("[", "(").replace("]", ")")
    
    def get_sym_dict(form: str, factor: float) -> dict:
        sym_dict = defaultdict(float)
        for match in re.finditer(r"([A-Z][a-z]*)\s*([-*\.e\d]*)", form):
            element = match[1]
            amt = 1.0
            if match[2].strip() != "":
                amt = float(match[2])
            sym_dict[element] += amt * factor
            form = form.replace(match.group(), "", 1)
        if form.strip():
            raise ValueError(f"Invalid formula segment: {form}")
        return sym_dict

    match = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
    while match:
        factor = float(match.group(2)) if match.group(2) != "" else 1.0
        unit_sym_dict = get_sym_dict(match.group(1), factor)
        expanded_sym = "".join(f"{el}{unit_sym_dict[el]}" for el in unit_sym_dict)
        formula = formula.replace(match.group(), expanded_sym, 1)
        match = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
    return get_sym_dict(formula, 1)

def cif_to_dict(path: str) -> dict:
    """
    Parse a CIF file and return a dictionary of its data.
    """
    if not os.path.isfile(path):
        print(f"File {path} not found!")
        return {}
    
    attributes_to_read = [
        '_chemical_formula_structural',
        '_chemical_formula_sum',
        '_chemical_name_structure_type',
        '_chemical_formula_weight',
        '_cell_length_a',
        '_cell_length_b',
        '_cell_length_c',
        '_cell_angle_alpha',
        '_cell_angle_beta',
        '_cell_angle_gamma',
        '_cell_volume',
        '_cell_formula_units_Z',
        '_space_group_IT_number',
        '_space_group_name_H-M_alt',
    ]
    
    data = defaultdict(lambda: None)
    with open(path, 'r') as f:
        lines = f.readlines()
        
        # Process "conditions" from (assumed) the third line in the file.
        conditions = lines[2].replace('#', "").lstrip().split()
        conditions = conditions[2] if len(conditions) > 3 else ""
        data['conditions'] = conditions
        
        ln = 0
        while ln < len(lines):
            line = lines[ln].lstrip()
            if not line.split():
                ln += 1
                continue
            parts = line.split()
            if parts[0] in attributes_to_read:
                next_line = lines[ln+1].lstrip() if ln+1 < len(lines) else ""
                if next_line.startswith('_') or next_line.startswith('loop_'):
                    data[parts[0]] = parts[1:]
                else:
                    line_data = ''
                    while ln+1 < len(lines) and not lines[ln+1].lstrip().startswith('_'):
                        ln += 1
                        line_data += lines[ln].strip().replace(';', '').replace(' ', '')
                    data[parts[0]] = line_data.strip()
            # Handle the loop block for atom site data
            if line.startswith('loop_') and ln+1 < len(lines) and lines[ln+1].lstrip().startswith('_atom_site'):
                site_data = []
                keys = []
                ln += 1
                while ln < len(lines) and lines[ln].lstrip().startswith('_atom_site'):
                    keys.append(lines[ln].lstrip().strip())
                    ln += 1
                # Read the actual atom site data lines until a new tag starts
                while ln < len(lines) and not lines[ln].lstrip().startswith('_'):
                    if lines[ln].strip():
                        site_data.append(dict(zip(keys, lines[ln].lstrip().split())))
                    ln += 1
                data['atom_site_data'] = site_data
            else:
                ln += 1
                
    data = format_cif_data(data)
    return dict(data)

def format_cif_data(cif_data: dict) -> dict:
    """
    Format CIF data: convert numeric fields, clean strings, and process atom site data.
    """
    numeric_attributes = [
        '_chemical_formula_weight',
        '_cell_length_a',
        '_cell_length_b',
        '_cell_length_c',
        '_cell_angle_alpha',
        '_cell_angle_beta',
        '_cell_angle_gamma',
        '_cell_volume',
        '_cell_formula_units_Z',
        '_space_group_IT_number',
    ]
    
    for k in numeric_attributes:
        if k in cif_data and cif_data[k] is not None:
            try:
                cif_data[k] = float(cif_data[k][0]) if isinstance(cif_data[k], list) else float(cif_data[k])
            except:
                cif_data[k] = -1.0
    
    string_attributes = [
        '_chemical_formula_sum',
        '_chemical_name_structure_type',
        '_space_group_name_H-M_alt',
        'conditions'
    ]
    
    for k in string_attributes:
        if k in cif_data and cif_data[k] is not None:
            if isinstance(cif_data[k], list):
                cif_data[k] = ''.join(cif_data[k]).replace("'", '').replace("~", "")
    
    # Process atom site data: convert coordinate entries to float and clean element symbols.
    site_data = []
    if cif_data.get('atom_site_data', None) is not None:
        for site in cif_data['atom_site_data']:
            sdict = {}
            for k, v in site.items():
                if k not in ['_atom_site_label', '_atom_site_type_symbol', '_atom_site_Wyckoff_symbol']:
                    try:
                        sdict[k] = float(v.split('(')[0])
                    except Exception:
                        sdict[k] = v
                elif k == "_atom_site_type_symbol":
                    symbol = v[:2]
                    if symbol[-1].isnumeric():
                        symbol = symbol[:-1]
                    sdict[k] = symbol
                else:
                    sdict[k] = v
            # Save the fractional coordinates as a list of floats.
            sdict['coordinates'] = [float(site.get('_atom_site_fract_x', 0)),
                                    float(site.get('_atom_site_fract_y', 0)),
                                    float(site.get('_atom_site_fract_z', 0))]
            site_data.append(sdict)
        cif_data['atom_site_data'] = site_data
    else:
        cif_data['atom_site_data'] = []
    
    # Parse the formula into a dict so that the number of elements can be computed.
    cif_data['formula'] = _parse_formula(cif_data.get('_chemical_formula_sum', ''))
    return cif_data

def find_sites_with_same_wyckoff_symbols(cifpath: str) -> dict:
    """
    Identify sites that share the same Wyckoff symbol (i.e. sites with multiple coordinate entries)
    and return those that occur more than once.
    """
    shared = {}
    sites = {}
    cif = cif_to_dict(cifpath)
    for site in cif['atom_site_data']:
        wyckoff = f"{int(site.get('_atom_site_symmetry_multiplicity', 1))}{site.get('_atom_site_Wyckoff_symbol', '')}"
        if wyckoff not in sites:
            sites[wyckoff] = [site['coordinates']]
        else:
            current_coords = sites[wyckoff]
            duplicate = any(np.allclose(coord, site['coordinates']) for coord in current_coords)
            if not duplicate:
                current_coords.append(site['coordinates'])
                sites[wyckoff] = current_coords
    for k, v in sites.items():
        if len(v) > 1:
            shared[k] = np.array(v)
    return shared

def get_site_label(site: dict, shared: dict) -> str:
    """
    Compute a unique site label based on its Wyckoff symbol and, if needed,
    an index to distinguish multiple entries.
    """
    wyckoff = f"{int(site.get('_atom_site_symmetry_multiplicity', 1))}{site.get('_atom_site_Wyckoff_symbol', '')}"
    if wyckoff in shared and len(shared):
        order = shared[wyckoff]
        ind = sorted([[i+1, np.linalg.norm(np.array(site['coordinates']) - order[i])]
                      for i in range(order.shape[0])],
                     key=lambda x: x[1])
        wyckoff = f'{wyckoff}{ind[0][0]}'
    return wyckoff

def get_data_row(cifpath: str, shared_wyckoffs: dict = None) -> dict:
    """
    Extract a row of data from the CIF file. The output dictionary will include fixed fields:
      - Filename
      - Formula
      - Entry prototype
      - Notes
      - Num Elements
    And for each atomic site, it provides a list in the form [element symbol(s), coordinates].
    """
    cif = cif_to_dict(cifpath)
    site_formula = []
    for site in cif['atom_site_data']:
        comp = site.get('_atom_site_type_symbol', '')
        # Use the shared Wyckoff information, if available, to generate a unique label.
        wyckoff = get_site_label(site, shared_wyckoffs) if shared_wyckoffs else site.get('_atom_site_Wyckoff_symbol', '')
        if float(site.get('_atom_site_occupancy', 1.0)) != 1.0:
            comp += str(site.get('_atom_site_occupancy', ''))
        site_formula.append([comp, wyckoff, site['coordinates']])
        
    # Combine entries that share the same site (if coordinates match)
    site_formula_r = {}
    for comp, wyckoff, coords in site_formula:
        if wyckoff not in site_formula_r:
            site_formula_r[wyckoff] = [comp, coords]
        else:
            existing_comp, existing_coords = site_formula_r[wyckoff]
            if np.allclose(coords, existing_coords):
                site_formula_r[wyckoff] = [existing_comp + comp, coords]
    
    data = {
        'Filename': os.path.basename(cifpath),
        'Formula': cif.get('_chemical_formula_sum', ''),
        'Entry prototype': cif.get('_chemical_name_structure_type', ''),
        'Notes': cif.get('conditions', ''),
        'Num Elements': len(cif.get('formula', {}))
    }
    for k, v in site_formula_r.items():
        data[k] = v  # Each value is [element symbol(s), coordinates]
    return data

def parse_cif_information(folder_path: str, output_folder: str = "outputs") -> None:
    """
    Given a folder path containing CIF files, parse each file and produce separate CSV files
    for each unique _chemical_name_structure_type found in the files. For each group:
      - The first row contains averaged fractional coordinates for each site (calculated from all files in that group).
      - The remaining rows hold the site element symbols.
    
    Output CSV files are saved in the specified output folder with file names based on the 
    _chemical_name_structure_type (sanitized to remove special characters).
    """
    # List all CIF files in the provided folder.
    cif_file_list = [os.path.join(folder_path, f) 
                     for f in os.listdir(folder_path) if f.endswith('.cif')]
    
    if not cif_file_list:
        raise ValueError("No CIF files provided.")
    
    # These keys remain fixed in the output.
    fixed_keys = ['Filename', 'Formula', 'Entry prototype', 'Notes', 'Num Elements']
    
    # Group CIF files by _chemical_name_structure_type.
    files_by_structure = defaultdict(list)
    for cif_file in cif_file_list:
        cif = cif_to_dict(cif_file)
        # Use a default key if the structure type is missing.
        structure = cif.get('_chemical_name_structure_type', 'unknown').strip()
        files_by_structure[structure].append(cif_file)
    
    # Ensure that the output folder exists.
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each structure type group individually.
    for structure, files in files_by_structure.items():
        group_rows = []
        group_sites_coord = {}   # To accumulate coordinate data for averaging.
        
        # Compute shared Wyckoff site information from the first CIF file in the group.
        shared = find_sites_with_same_wyckoff_symbols(files[0])
        # Use one sample row to determine which site keys are present.
        sample_row = get_data_row(files[0], shared)
        all_sites = [k for k in sample_row.keys() if k not in fixed_keys]
        # Sort the site keys (adjust the sorting as needed)
        all_sites = sorted(all_sites, key=lambda k: (int(re.split('[a-zA-Z]', k)[0]), re.split('[0-9]', k)[0]))
        
        # Initialize the coordinates collector for each site.
        for site in all_sites:
            group_sites_coord[site] = []
        
        # Process each file in this group.
        for cif_file in files:
            row = get_data_row(cif_file, shared)
            for site in all_sites:
                if site in row:
                    # row[site] is in the form [element symbol, coordinates]
                    coord = row[site][1]
                    group_sites_coord[site].append(coord)
                    # Replace the entry with only the element symbol.
                    row[site] = row[site][0]
                else:
                    group_sites_coord[site].append([np.nan, np.nan, np.nan])
                    row[site] = ""
            group_rows.append(row)
        
        # Compute the average (and standard deviation, if significant) of coordinates for each site.
        pos_row = {key: "" for key in fixed_keys + all_sites}
        for site in all_sites:
            coords_array = np.array(group_sites_coord[site], dtype=float)
            avg_coords = []
            for i in range(3):
                mean_val = coords_array[:, i].mean()
                std_val = coords_array[:, i].std()
                if std_val > 0.001:
                    avg_coords.append(f"{mean_val:.3f}({int(round(std_val, 3)*1000)})")
                else:
                    avg_coords.append(f"{mean_val:.3f}")
            pos_row[site] = " ".join(avg_coords)
        
        # Insert the row of averaged coordinates at the top of the data.
        group_rows.insert(0, pos_row)
        df = pd.DataFrame(group_rows)
        
        # Sanitize the structure string so it is safe for use as a filename.
        file_name = "".join(c for c in structure if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(" ", "_")
        if not file_name:
            file_name = "unknown"
        output_csv = os.path.join(output_folder, f"{file_name}.csv")
        
        df.to_csv(output_csv, index=False)
    return df
    
    print("CSV files created in:", output_folder)