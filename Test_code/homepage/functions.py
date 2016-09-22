from query_engine import QueryEngine

from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.phasediagram.maker import PhaseDiagram
from pymatgen.phasediagram.analyzer import PDAnalyzer
from pymatgen.core.ion import Ion
from pymatgen import Element
from pymatgen.analysis.pourbaix.entry import PourbaixEntry, IonEntry
from pymatgen.analysis.pourbaix.maker import PourbaixDiagram
from pymatgen.analysis.pourbaix.plotter import PourbaixPlotter
from pymatgen.analysis.pourbaix.analyzer import PourbaixAnalyzer

import rpy2 as R
import pandas as pd

import itertools

import ast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from monty.serialization import loadfn
import os
import sys

import boto

from plotter import BSPlotter

from settings import BASE_DIR, DEBUG


DB_CONFIG = os.path.join(BASE_DIR, 'dbauth.yaml')
CREDS = loadfn(DB_CONFIG)
USR = CREDS['username']
PWD = CREDS['password']

ION_DATA = loadfn(os.path.join(BASE_DIR, 'ions.yaml'))
END_MEMBERS = loadfn(os.path.join(BASE_DIR, 'end_members.yaml'))
ION_COLORS = loadfn(os.path.join(BASE_DIR, 'ion_colors.yaml'))

if DEBUG:
    host = '10.5.46.101'
else:
    host = 'localhost'
    conn = boto.connect_s3(os.environ['AWS_ACCESS_KEY_ID'],
                           os.environ['AWS_SECRET_ACCESS_KEY'])
    bucket = conn.get_bucket('elasticbeanstalk-us-east-1-262801263182')

def get_entries_in_chemsys(chemsys, collection):
    """
    Survey the vasp database on Hydrogen for entries in a chemical
    system. e.g. Li-Fe-O will return all Li-O, Fe-O, Li-Fe, Li-Fe-O
    compounds.
    """

    query_engine = QueryEngine(host=host, port=27017,
                               database='vasp', collection=collection,
                               user=USR, password=PWD)

    return query_engine.get_entries_in_system(chemsys)

def get_nist_data(query={'element':'Al', 'code': 'VASP', 'exchange':'pbe', 'propertry':'a0'}):
    """
    uses MongoClient to connect to Hydrogen database and the NIST collection
    """

    mongo_engine = MongoClient(host=host, port=27017)
    db = mongo_engine['vasp']
    if db.authenticate(user=USR, password=PWD):
        collection = db.NIST
    return collection.find(query)

def get_properties(chemsys, collection, additional_criteria={}):
    """
    Survey the vasp database on Hydrogen for materials in the chemsys.
    """

    if '-' in chemsys:
        is_formula = False
        elements = chemsys.split('-')
        options = [''.join(o) for o in itertools.permutations(
            [i for i in elements] + ['-'] * (len(elements) - 2))]
    else:
        is_formula = True
        formula = Composition(chemsys).reduced_formula

    query_engine = QueryEngine(host=host, port=27017,
                               database='vasp', collection=collection,
                               user=USR, password=PWD)
    if is_formula:
        criteria = {'pretty_formula': formula}
        criteria.update(additional_criteria)
        results = query_engine.query(criteria=criteria)
    else:
        criteria = {'chemsys': {'$in': options}}
        criteria.update(additional_criteria)
        results = query_engine.query(criteria=criteria)

    return results[0]


def get_properties_by_material_id(material_id, collection,
                                  additional_criteria={}):
    """
    Survey the vasp database on Hydrogen for a material by id.
    """


    query_engine = QueryEngine(host=host, port=27017,
                               database='vasp', collection=collection,
                               user=USR, password=PWD)
    if is_formula:
        criteria = {'pretty_formula': formula}
        criteria.update(additional_criteria)
        results = query_engine.query(criteria=criteria)
    else:
        criteria = {'chemsys': {'$in': options}}
        criteria.update(additional_criteria)
        results = query_engine.query(criteria=criteria)

    return results[0]


def get_all_properties(query, collection, run_type):
    """
    Survey the vasp database on Hydrogen for materials in the chemsys.
    """

    query_engine = QueryEngine(host=host, port=27017,
                               database='vasp', collection=collection,
                               user=USR, password=PWD)
    if '{' in query:
        literal_query = ast.literal_eval(query)
        literal_query.update({'run_type': run_type})
        results = query_engine.query(criteria=literal_query)
    elif 'mw-' in query:
        results = query_engine.query(criteria={'material_id': query,
                                               'run_type': run_type})
    elif '-' in query:
        if query[-1] == '-':
            query = query[:-1]
        elements = query.split('-')
        options = [''.join(o) for o in itertools.permutations(
            [i for i in elements] + ['-'] * (len(elements) - 1))]
        results = query_engine.query(criteria={'chemsys': {'$in': options},
                                               'run_type': run_type})
    else:
        formula = Composition(query).reduced_formula
        results = query_engine.query(criteria={'pretty_formula': formula,
                                               'run_type': run_type})

    return results


def get_hull_distance(comp, energy, collection):
    """
    comp, energy: composition & energy of your compound.
    """

    my_entry = ComputedEntry(comp, energy)
    entries = get_entries_in_chemsys(
        [elt.symbol for elt in comp.elements], collection)

    pda = PDAnalyzer(PhaseDiagram(entries))
    competition = pda.get_decomp_and_e_above_hull(
        my_entry, allow_negative=True)
    competing_species = [
        e.composition.reduced_formula for e in competition[0]]
    delta_h = -1000 * round(competition[1], 3)
    return '{} --> {} ({} meV/atom)'.format(
        my_entry.composition.reduced_formula, ' + '.join(competing_species),
        delta_h)


def get_m2ax_composition(formula):
    """
    Accepts a general M-A-X chemsys and returns a M2AX formula.
    """
    formula = str(formula)
    if len(formula) != 0:
        m_atoms = ['Sc', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta']
        a_atoms = ['Al', 'Si', 'P', 'S', 'Ga', 'Ge', 'As', 'Cd', 'In', 'Sn',
                   'Tl', 'Pb']
        x_atoms = ['C', 'N']
        comp_as_dict = Composition(formula).as_dict()
        m_elts = []
        a_elts = []
        x_elts = []
        for elt in comp_as_dict:
            if elt in m_atoms and elt not in a_atoms and elt not in x_atoms:
                m_elts.append(elt)
            elif elt in a_atoms and elt not in x_atoms:
                a_elts.append(elt)
            elif elt in x_atoms:
                x_elts.append(elt)
        if len(m_elts) == 1 and len(a_elts) == 1 and len(x_elts) == 1:
            comp_as_dict[m_elts[0]] = 4
            comp_as_dict[a_elts[0]] = 2
            comp_as_dict[x_elts[0]] = 2
        elif len(m_elts) == 1 and len(a_elts) == 2 and len(x_elts) == 1:
            comp_as_dict[m_elts[0]] = 4
            comp_as_dict[x_elts[0]] = 2
        elif len(m_elts) == 1 and len(a_elts) == 1 and len(x_elts) == 2:
            comp_as_dict[m_elts[0]] = 4
            comp_as_dict[a_elts[0]] = 2
        elif len(m_elts) == 1 and len(a_elts) == 2 and len(x_elts) == 2:
            comp_as_dict[m_elts[0]] = 4
        elif len(m_elts) == 2 and len(a_elts) == 2 and len(x_elts) == 1:
            comp_as_dict[m_elts[0]] = 2
            comp_as_dict[m_elts[1]] = 2
            comp_as_dict[x_elts[0]] = 2
        elif len(m_elts) == 2 and len(a_elts) == 1 and len(x_elts) == 2:
            comp_as_dict[m_elts[0]] = 2
            comp_as_dict[m_elts[1]] = 2
            comp_as_dict[a_elts[0]] = 2
        elif len(m_elts) == 2 and len(a_elts) == 1 and len(x_elts) == 1:
            comp_as_dict[m_elts[0]] = 2
            comp_as_dict[m_elts[1]] = 2
            comp_as_dict[a_elts[0]] = 2
            comp_as_dict[x_elts[0]] = 2
        elif len(m_elts) == 2 and len(a_elts) == 2 and len(x_elts) == 2:
            comp_as_dict[m_elts[0]] = 2
            comp_as_dict[m_elts[1]] = 2

    return Composition(comp_as_dict)


def loadbulkresults(chemsys, spacegroup, ionic_concentration):
    """
    Call the get_properties() method for a bulk material whose
    formula matches the one given. Returns a context dictionary
    containing the reduced formula, a & c lattice parameters, and
    energy per atom.
    """

    if len(chemsys) == 0:
        results = {'redform': '', 'formation_energy': '',
                   'alp': '', 'blp': '', 'evpa': ''}
    else:
        try:
            properties = get_properties(
                chemsys, 'MAX_phases',
                additional_criteria={'run_type': 'GGA',
                                     'spacegroup.symbol': spacegroup})
        #            hse_properties = get_properties(
        #                chemsys, 'MAX_phases',
        #                additional_criteria={'run_type': 'HF',
        #                                     'spacegroup.symbol': spacegroup})

            if len(properties) == 0:
                results = {'redform': '', 'alp': '', 'blp': '',
                           'spacegroup': '', 'structure': '',
                           'diagram_path': '', 'formation_energy': '',
                           'err_msg': 'Entry not found.'}
            else:
        #            upload_bands_diagram(hse_properties)
                alp = round(properties['output']['crystal']['lattice']['a'], 3)
                clp = round(properties['output']['crystal']['lattice']['c'], 3)
                sgp = properties['spacegroup']['symbol']
        #            bgp = round(hse_properties['analysis']['bandgap'], 3)
                structure = Structure.from_dict(
                    properties['output']['crystal'])
                composition = structure.composition
                energy = properties['output']['final_energy']
                analyzer = SpacegroupAnalyzer(structure)
                structure = analyzer.get_refined_structure()
                structure.make_supercell([2, 2, 2])

                formation_energy = get_hull_distance(composition, energy,
                                                     'MAX_phases')
                reaction = formation_energy.split()
                if reaction[0] == reaction[2]:
                    formation_energy = 'Thermodynamically Stable'

                incar_dict = {}
                supported_tags = ['AGGAC', 'EDIFF', 'GGA', 'IBRION', 'ISIF',
                                  'ISMEAR', 'LCHARG', 'LREAL', 'LUSE_VDW',
                                  'NSW', 'PARAM1', 'PARAM2', 'PREC', 'ENMAX',
                                  'SIGMA', 'LVTOT', 'LVHAR', 'IALGO']
                all_parameters = (
                    properties['calculations'][0]['input']['parameters'])

                for tag in supported_tags:
                    incar_dict[tag] = all_parameters[tag]

                kpoints_dict = properties['calculations'][0]['input']['kpoints']
                if kpoints_dict['generation_style'] == 'Monkhorst-Pack':
                    kpoints_dict['generation_style'] = 'Monkhorst'

                results = {'redform': properties['pretty_formula'],
                           'alp': 'a: {} Angstroms'.format(alp),
                           'blp': 'c: {} Angstroms'.format(clp),
                           'spacegroup': 'Space Group: {}'.format(sgp),
                           'structure': structure,
                           'bandgap': 'not yet calculated',
                           'formation_energy': formation_energy,
                           'incar_dict': incar_dict,
                           'kpoints_dict': kpoints_dict,
                           'composition': composition,
                           'energy': energy,
                           'err_msg': ''}
        except Exception as e:
            print e
            results = {'redform': '', 'alp': '', 'blp': '', 'spacegroup': '',
                       'structure': '', 'formation_energy': '',
                       'err_msg': 'Invalid formula.'}

    return results


def get_xyz_string_structure(structure):
    """
    Given a pymatgen Structure object, returns an xyz string. Useful for
    Jmol INLINE representation, which is actually pretty annoying.
    """

    xyz_structure = [str(structure.num_sites),
                     structure.composition.reduced_formula]
    for site in structure.sites:
        element = site._species.reduced_formula.replace('2', '')
        atom = '{} {} {} {}'.format(element, str(site.x), str(site.y),
                                    str(site.z))
        xyz_structure.append(atom)

    return '+'.join(xyz_structure)


def load2dresults(material_id, ionic_concentration):
    """
    Call the get_properties() method for a 2D material whose
    material_id matches the one given. Returns a context
    dictionary containing the reduced formula, a & b lattice
    parameters, and band gap.
    """

#    try:
    query_engine = QueryEngine(host=host, port=27017,
                               database='vasp', collection='mat2d',
                               user=USR, password=PWD)
    pbe_query = [r for r in query_engine.query(
        criteria={'material_id': material_id, 'run_type': 'GGA'})]

    normal_calculations = [r for r in pbe_query if r['calculations'][0]\
        ['input']['kpoints']['generation_style'] != 'Line_mode']

    # The default band structure is PBE, unless
    # an HSE calculation is found.

    pbe_results = [r for r in pbe_query if r['calculations']\
        [0]['input']['kpoints']['generation_style'] == 'Line_mode']

    # This should not happen, but it can. No entry is
    # found for the normal relaxation, but there is
    # one for the PBE band structure.
    if len(normal_calculations):
        properties = normal_calculations[0]
    elif len(pbe_results):
        properties = pbe_results[0]

    if len(pbe_results):
        band_structure_properties = pbe_results[0]
        theory_level = 'PBE'

    hse_query = query_engine.query(
        criteria={'material_id': material_id, 'run_type': 'HF'})

    if len(hse_query):
        band_structure_properties = hse_query[0]
        theory_level = 'HSE'

    if len(properties) == 0:
        results = {'redform': '', 'alp': '', 'blp': '',
                   'spacegroup': '', 'structure': '',
                   'diagram_path': '', 'theory_level': '',
                   'err_msg': 'Entry not found.'}
    else:
        bands = upload_bands_diagram(band_structure_properties, theory_level)
        alp = round(properties['output']['crystal']['lattice']['a'], 3)
        blp = round(properties['output']['crystal']['lattice']['b'], 3)
        sgp = properties['spacegroup']['symbol']
        bgp = round(band_structure_properties['analysis']['bandgap'], 3)
        structure = Structure.from_dict(
            properties['output']['crystal'])
        composition = structure.composition
        energy = properties['output']['final_energy']
        analyzer = SpacegroupAnalyzer(structure)
        structure = analyzer.get_refined_structure()
#                formation_energy = get_hull_distance(composition, energy,
#                                                     'MAX_phases')

        incar_dict = {}
        supported_tags = ['AGGAC', 'EDIFF', 'GGA', 'IBRION', 'ISIF',
                          'ISMEAR', 'LCHARG', 'LREAL', 'LUSE_VDW',
                          'NSW', 'PARAM1', 'PARAM2', 'PREC', 'ENMAX',
                          'SIGMA', 'LVTOT', 'LVHAR', 'IALGO']
        all_parameters = (
            properties['calculations'][0]['input']['parameters'])

        for tag in supported_tags:
            incar_dict[tag] = all_parameters[tag]

        kpoints_dict = properties['calculations'][0]['input']['kpoints']
        if kpoints_dict['generation_style'] == 'Monkhorst-Pack':
            kpoints_dict['generation_style'] = 'Monkhorst'

        needs_shift = False

        ### Determine vacuum axis and make 2D supercell.
        if structure.lattice.a == max(structure.lattice.abc):
            translation = SymmOp.from_rotation_and_translation(
                translation_vec=(structure.lattice.a/2, 0, 0)
                )
            for site in structure.sites:
                if site._fcoords[0] > 0.9 or site._fcoords[0] < 0:
                    needs_shift = True
            if needs_shift:
                structure.apply_operation(translation)
            structure.make_supercell([1, 6, 6])
        elif structure.lattice.b == max(structure.lattice.abc):
            translation = SymmOp.from_rotation_and_translation(
                translation_vec=(0, structure.lattice.b/2, 0)
                )
            for site in structure.sites:
                if site._fcoords[1] > 0.9 or site._fcoords[1] < 0:
                    needs_shift = True
            if needs_shift:
                structure.apply_operation(translation)
            structure.make_supercell([6, 1, 6])
        else:
            translation = SymmOp.from_rotation_and_translation(
                translation_vec=(0, 0, structure.lattice.c/2)
                )
            for site in structure.sites:
                if site._fcoords[2] > 0.9 or site._fcoords[2] < 0:
                    needs_shift = True
            if needs_shift:
                structure.apply_operation(translation)
            structure.make_supercell([6, 6, 1])

        if bgp and band_structure_properties['analysis']['is_gap_direct']:
            bandgap = '(direct) {}'.format(bgp)
        elif bgp:
            bandgap = '(indirect) {}'.format(bgp)
        else:
            bandgap = bgp
        results = {'redform': properties['pretty_formula'],
                   'alp': 'a: {} Angstroms'.format(alp),
                   'blp': 'b: {} Angstroms'.format(blp),
                   'spacegroup': 'Space Group: {}'.format(sgp),
                   'bandgap': '{} eV'.format(bandgap),
                   'structure': structure,
                   'incar_dict': incar_dict,
                   'kpoints_dict': kpoints_dict,
                   'composition': composition,
                   'energy': energy,
                   'theory_level': theory_level,
                   'bands_plot': bands,
                   'err_msg': ''}

#    except Exception as e:
#        exc_type, exc_obj, exc_tb = sys.exc_info()
#        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#        print e, exc_type, fname, exc_tb.tb_lineno

#        results = {'redform': '', 'alp': '', 'blp': '', 'spacegroup': '',
#                   'structure': '', 'theory_level': '',
#                   'err_msg': 'Invalid formula.'}

    return results


def get_perovskite_formula(formula):
    """
    Accepts a general 'A-B' or 'A-B-O' formula and returns an 'ABO3'
    formula.
    """
    redform = str(Composition(formula).reduced_formula)
    comp_as_dict = Composition(redform).as_dict()
    comp_as_dict.update({'O': 3})
    return str(Composition(comp_as_dict).reduced_formula)


def loadpiezoresults(formula):
    """
    Call the get_properties() method for a Piezoelectric object whose
    'name' matches the given formula. Returns a context dictionary
    containing the reduced formula, c by a ratio, and energy per atom.
    """
    if len(formula) == 0:
        results = {'redform': '', 'c_by_a': '', 'evpa': '', 'bandgap': '',
                   'err_msg': ''}
    else:
        try:
            redform = get_perovskite_formula(formula)
            properties = get_properties(redform, 'mat2d')
            if len(properties) == 0:
                results = {'redform': '', 'c_by_a': '', 'evpa': '',
                           'err_msg': 'Entry not found.'}
            else:
                c_by_a = round(properties['output']['crystal']['lattice']['c']
                               / properties['output']['crystal']['lattice']
                               ['a'],
                               3)
                evpa = round(properties['output']['final_energy_per_atom'], 3)
                bgp = round(properties['analysis']['bandgap'], 3)
                if bgp and properties['analysis']['is_gap_direct']:
                    bandgap = '(direct) {}'.format(bgp)
                elif bgp:
                    bandgap = '(indirect) {}'.format(bgp)
                else:
                    bandgap = bgp
                results = {'redform': 'Formula: {}'.format(redform),
                           'c_by_a': 'c/a: {}'.format(c_by_a),
                           'evpa': 'Energy per atom: {}'.format(evpa),
                           'bandgap': 'Band Gap: {} eV'.format(bandgap),
                           'err_msg': ''}
        except Exception as e:
            print e
            results = {'redform': '', 'c_by_a': '', 'evpa': '', 'bandgap': '',
                       'err_msg': 'Invalid formula.'}
    return results

def loadNIST(filename=None, query={}):
    """
    loads all NIST data from the NIST collection on 
    mongoDB 
    Filename provided as temporary testing in place
    of DB 
    converts database/csv data to pandas dataframe
    """
    if filename:
       dat = pd.read_csv('./temp_data'+os.sep+filename)
    elif query:
       dat = pd.from_json(get_nist_data(query))   
    return dat

def filterNIST_data(df, tags):
    """
    df: starting dataframe
    tags: dict of identifiers ['column_name':identifying field]
    """
    filtered_df = df
    for c,i in tags.items():
        filtered_df = filtered_df[filtered_df[c]==i]
    return filtered_df

def get_database_populations():
    """
    Counts the number of entries in each collection on Hydrogen so they
    can be reported on the home page.
    """

    populations = dict()
    try:
        for collection in ['mat2d', 'MAX_phases']:
            query_engine = QueryEngine(host=host, port=27017,
                                       database='vasp', collection=collection,
                                       user=USR, password=PWD)
            populations[collection] = len(query_engine.query(criteria={}))

        populations['mat2d'] /= 2  # Don't double count HSE runs.
#        populations['MAX_phases'] -= 99  # Remove duplicates.

    except Exception as e:
        print e
        for collection in ['mat2d', 'MAX_phases']:
            populations[collection] = 0

    for collection in populations:
        populations[collection] = '0' * (5-len(str(populations[collection])))\
            + str(populations[collection])

    return populations


def contains_entry(entry_list, entry):
    """
    Helper function for filtering duplicate entries.
    """

    for ent in entry_list:
        if (ent.entry_id == entry.entry_id
            or (abs(entry.energy_per_atom - ent.energy_per_atom) < 1e-6
                and (entry.composition.reduced_formula ==
                     ent.composition.reduced_formula)
                )):
            return True


def remove_z_kpoints(filename='KPOINTS'):
    """
    Helper function to strip all paths from a linemode KPOINTS that
    include a z-component, since these are not relevant for 2D
    materials.
    """

    kpoint_lines = open(filename).readlines()
    with open(filename, 'w') as kpts:
        for line in kpoint_lines[:4]:
            kpts.write(line)
        i = 4
        while i < len(kpoint_lines):
            if (
                not float(kpoint_lines[i].split()[2]) and
                not float(kpoint_lines[i+1].split()[2])
                    ):
                kpts.write(kpoint_lines[i])
                kpts.write(kpoint_lines[i+1])
                kpts.write('\n')
            i += 3


def upload_pourbaix_diagram(composition, energy, material_id,
                            ion_concentration=1e-6):
    """
    Plot a pourbaix diagram based on a composition, energy, and
    ion_concentration (M). `composition` should be a pymatgen
    Composition object, and energy should be the absolute energy
    corresponding to the composition.
    """

    cmpd = ComputedEntry(composition, energy)

    # Define the chemsys that describes the 2D compound.
    chemsys = ['O', 'H'] + [elt.symbol for elt in composition.elements
                            if elt.symbol not in ['O', 'H']]

    # Experimental ionic energies
    # See ions.yaml for ion formation energies and references.
    exp_dict = ION_DATA['ExpFormEnergy']
    ion_correction = ION_DATA['IonCorrection']

    # Pick out the ions pertaining to the 2D compound.
    ion_dict = dict()
    for elt in chemsys:
        if elt not in ['O', 'H'] and exp_dict[elt]:
            ion_dict.update(exp_dict[elt])

    elements = [Element(elt) for elt in chemsys if elt not in ['O', 'H']]

    # Calculate formation energy of the compound from its end
    # members
    form_energy = cmpd.energy
    for elt in composition.as_dict():
        form_energy -= END_MEMBERS[elt] * cmpd.composition[elt]

    # Convert the compound entry to a pourbaix entry.
    # Default concentration for solid entries = 1
    pbx_cmpd = PourbaixEntry(cmpd)
    pbx_cmpd.g0_replace(form_energy)
    pbx_cmpd.reduced_entry()

    # Add corrected ionic entries to the pourbaix diagram
    # dft corrections for experimental ionic energies:
    # Persson et.al PHYSICAL REVIEW B 85, 235438 (2012)
    pbx_ion_entries = list()

    # Get PourbaixEntry corresponding to each ion.
    # Default concentration for ionic entries = 1e-6
    # ion_energy = ion_exp_energy + ion_correction * factor
    # where factor = fraction of element el in the ionic entry
    # compared to the reference entry
    for elt in elements:
        for key in ion_dict:
            comp = Ion.from_formula(key)
            if comp.composition[elt] != 0:
                factor = comp.composition[elt]
                energy = ion_dict[key]
                pbx_entry_ion = PourbaixEntry(IonEntry(comp, energy))
                pbx_entry_ion.correction = ion_correction[elt.symbol]\
                    * factor
                pbx_entry_ion.conc = ion_concentration
                pbx_entry_ion.name = key
                pbx_ion_entries.append(pbx_entry_ion)

    # Generate and plot Pourbaix diagram
    # Each bulk solid/ion has a free energy g of the form:
    # g = g0_ref + 0.0591 * log10(conc) - nO * mu_H2O +
    # (nH - 2nO) * pH + phi * (-nH + 2nO + q)

    all_entries = [pbx_cmpd] + pbx_ion_entries

    pourbaix = PourbaixDiagram(all_entries)

    # Analysis features
    panalyzer = PourbaixAnalyzer(pourbaix)

    plotter = PourbaixPlotter(pourbaix)
    plot = plotter.get_pourbaix_plot(limits=[[0, 14], [-2, 2]],
                                     label_domains=True)
    fig = plot.gcf()
    ax1 = fig.gca()

    # Add coloring to highlight the stability region for the 2D
    # material, if one exists.
    stable_entries = plotter.pourbaix_plot_data(
        limits=[[0, 14], [-2, 2]])[0]

    for entry in stable_entries:
        if entry == pbx_cmpd:
            col = plt.cm.Blues(0)
        else:
            col = plt.cm.jet(float(
                ION_COLORS[entry.composition.reduced_formula]))

        vertices = plotter.domain_vertices(entry)
        patch = Polygon(vertices, closed=True, fill=True, color=col)
        ax1.add_patch(patch)

    fig.set_size_inches((11.5, 9))
    plot.tight_layout(pad=1.09)

    # Save plot
    plot.gca().set_xticklabels([int(t) for t in plot.gca().get_xticks()],
                               family='serif', size=30)
    plot.gca().set_yticklabels(plot.gca().get_yticks(), family='serif', size=30)
    plot.gca().set_xlabel(plot.gca().get_xlabel(), family='serif', size=30)
    plot.gca().set_ylabel(plot.gca().get_ylabel(), family='serif', size=30)

    if not DEBUG:
        plot.savefig('pourbaix.png', transparent=True)
        k = bucket.new_key(
            'static/pourbaix/{}.png'.format(material_id))
        k.set_contents_from_filename(os.path.join(os.getcwd(),
                                                  'pourbaix.png'))
        plot.savefig('pourbaix.pdf', transparent=True)
        k = bucket.new_key(
            'static/pourbaix/{}.pdf'.format(material_id))
        k.set_contents_from_filename(os.path.join(os.getcwd(),
                                                  'pourbaix.pdf'))
        os.system('rm pourbaix.png; rm pourbaix.pdf')

    return plot


def upload_bands_diagram(query_result, theory_level):
    """
    Plots a band structure (Only for 2D HSE bandstructure
    calculations! Can easily be modified for 3D or pbe
    calculations.) and uploads it to the S3 server.
    `query_result` should be a result obtained by the
    QueryEngine.query() method, or by get_properties()
    above.
    """

    r = query_result

    structure = Structure.from_dict(r['calculations'][0]['output']['crystal'])

    lattice = structure.lattice

    efermi = r['calculations'][0]['output']['efermi']

    # `raw_eigenvals` are not formatted correctly. Below, we
    # re-format them and save them as `eigenvals`
    raw_eigenvals = r['calculations'][0]['output']['eigenvalues']
    eigenvals = {Spin.up: []}

    # Add spin-polarization, if it's there in the calculation.
    if '-1' in raw_eigenvals['0']:
        eigenvals.update({Spin.down: []})

    n_bands = len(raw_eigenvals['0']['1'])
    n_kpoints = max([int(kpt) for kpt in raw_eigenvals]) + 1
    kpts_dict = r['calculations'][0]['input']['kpoints']

    # Everything below is basically to get the kpoints in their
    # proper format, with labels, etc. This is trickier than it
    # needs to be.
    kpath = HighSymmKpath(structure)
    Kpoints.automatic_linemode(0, kpath).write_file('KPOINTS')
    remove_z_kpoints()
    linemode_lines = open('KPOINTS').readlines()
    os.system('rm KPOINTS')

    n_nodes = (len(linemode_lines) - 4) / 3

    if theory_level == 'HSE':
        for i in range(n_bands):
            eigenvals[Spin.up].append([])
            if Spin.down in eigenvals:
                eigenvals[Spin.down].append([])
        for j in range(n_kpoints):
            if kpts_dict['kpts_weights'][j] == 0:
                for k in range(len(raw_eigenvals[str(j)]['1'])):
                    eigenvals[Spin.up][k].append(
                        raw_eigenvals[str(j)]['1'][k][0])
                    if Spin.down in eigenvals:
                        eigenvals[Spin.down][k].append(
                            raw_eigenvals[str(j)]['-1'][k][0])

        # We need to know how many divs were used between high symmetry
        # kpoints (nodes) in the original calculation.
        n_path_kpoints = len([weight for weight in kpts_dict['kpts_weights']
                              if weight == 0])
        n_divs = (n_path_kpoints - n_nodes) / n_nodes

    else:
        for i in range(n_bands):
            eigenvals[Spin.up].append([])
            eigenvals[Spin.down].append([])
        for j in range(n_kpoints):
            for k in range(len(raw_eigenvals[str(j)]['1'])):
                eigenvals[Spin.up][k].append(raw_eigenvals[str(j)]['1'][k][0])
                eigenvals[Spin.down][k].append(
                    raw_eigenvals[str(j)]['-1'][k][0])
        n_divs = n_kpoints / n_nodes

    kpts = []
    labels = {}
    i = 4

    # Re-create and label all the kpoints by hand.
    while i < len(linemode_lines):
        start_kpt = linemode_lines[i].split()
        end_kpt = linemode_lines[i+1].split()
        increments = [
            (float(end_kpt[0]) - float(start_kpt[0])) / n_divs,
            (float(end_kpt[1]) - float(start_kpt[1])) / n_divs,
            (float(end_kpt[2]) - float(start_kpt[2])) / n_divs
        ]

        kpt = [float(coord) for coord in start_kpt[:3]]
        kpts.append(kpt)

        labels[start_kpt[4]] = kpt
        for n in range(1, n_divs):
            kpt = [float(start_kpt[0]) + increments[0] * n,
                   float(start_kpt[1]) + increments[1] * n,
                   float(start_kpt[2]) + increments[2] * n]
            kpts.append(kpt)
        kpt = [float(coord) for coord in end_kpt[:3]]
        if theory_level == 'HSE':
            kpts.append(kpt)

        labels[end_kpt[4]] = kpt
        i += 3

    bs = BandStructureSymmLine(kpts, eigenvals, lattice, efermi,
                               labels_dict=labels)

    plot = BSPlotter(bs).get_plot()

    ax = plot.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plot.text(xlim[1], ylim[0] - 0.1, r'$\mathrm{%s}$' % end_kpt[4],
              horizontalalignment='center', verticalalignment='top',
              size=32)

    plot.savefig('bands.pdf', transparent=True)
    plot.savefig('bands.png', transparent=True)

    if not DEBUG:
        k = bucket.new_key('static/bands/{}.png'.format(r['material_id']))
        k.set_contents_from_filename(os.path.join(os.getcwd(),
                                                  'bands.png'))
        k = bucket.new_key('static/bands/{}.pdf'.format(r['material_id']))
        k.set_contents_from_filename(os.path.join(os.getcwd(),
                                                  'bands.pdf'))
        os.system('rm bands.png; rm bands.pdf')

    return plot
