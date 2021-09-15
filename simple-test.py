# This code is a part of Qiskit
# © Copyright IBM 2017, 2021.

# This code is licensed under the Apache License, Version 2.0. 

# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import Aer

from qiskit_nature.drivers import PySCFDriver, UnitsType, Molecule
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization import QubitConverter

from qiskit_nature.transformers import ActiveSpaceTransformer
from qiskit_nature.algorithms import GroundStateEigensolver, BOPESSampler
from qiskit.algorithms import NumPyMinimumEigensolver

from qiskit.utils import QuantumInstance

from qiskit_nature.circuit.library.ansatzes import UCCSD
from qiskit_nature.circuit.library.initial_states import HartreeFock
from qiskit.circuit.library import TwoLocal

from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA

from functools import partial as apply_variation_to_atom_pair
import numpy as np
import matplotlib.pyplot as plt

################################################# FUNCTION DEFINITION
def construct_hamiltonian_solve_ground_state(
    molecule,
    num_electrons=2,
    num_molecular_orbitals=2,
    chemistry_inspired=True,
    hardware_inspired_trial=None,
    vqe=True,
    perturbation_steps=np.linspace(-1, 1, 3),
):
    """Creates fermionic Hamiltonion and solves for the energy surface.

    Args:
        molecule (Union[qiskit_nature.drivers.molecule.Molecule, NoneType]): The molecule to simulate.
        num_electrons (int, optional): Number of electrons for the `ActiveSpaceTransformer`. Defaults to 2.
        num_molecular_orbitals (int, optional): Number of electron orbitals for the `ActiveSpaceTransformer`. Defaults to 2.
        chemistry_inspired (bool, optional): Whether to create a chemistry inspired trial state. `hardware_inspired_trial` must be `None` when used. Defaults to True.
        hardware_inspired_trial (QuantumCircuit, optional): The hardware inspired trial state to use. `chemistry_inspired` must be False when used. Defaults to None.
        vqe (bool, optional): Whether to use VQE to calculate the energy surface. Uses `NumPyMinimumEigensolver if False. Defaults to True.
        perturbation_steps (Union(list,numpy.ndarray), optional): The points along the degrees of freedom to evaluate, in this case a distance in angstroms. Defaults to np.linspace(-1, 1, 3).

    Raises:
        RuntimeError: `chemistry_inspired` and `hardware_inspired_trial` cannot be used together. Either `chemistry_inspired` is False or `hardware_inspired_trial` is `None`.

    Returns:
        qiskit_nature.results.BOPESSamplerResult: The surface energy as a BOPESSamplerResult object.
    """
    # Verify that `chemistry_inspired` and `hardware_inspired_trial` do not conflict
    if chemistry_inspired and hardware_inspired_trial is not None:
        raise RuntimeError(
            (
                "chemistry_inspired and hardware_inspired_trial"
                " cannot both be set. Either chemistry_inspired"
                " must be False or hardware_inspired_trial must be none."
            )
        )

    # Step 1 including refinement, passed in

    # Step 2a
    molecular_orbital_maker = PySCFDriver(
        molecule=molecule, unit=UnitsType.ANGSTROM, basis="sto3g"
    )

    # Refinement to Step 2a
    # They have helped you split the quantum and classical bit
    split_into_classical_and_quantum = ActiveSpaceTransformer(
        num_electrons=num_electrons, num_molecular_orbitals=num_molecular_orbitals
    )

    fermionic_hamiltonian = ElectronicStructureProblem(
        molecular_orbital_maker, [split_into_classical_and_quantum]
    )
    fermionic_hamiltonian.second_q_ops()

    # Step 2b
    map_fermions_to_qubits = QubitConverter(JordanWignerMapper())

    # Step 3a
    if chemistry_inspired:
        molecule_info = fermionic_hamiltonian.molecule_data_transformed
        num_molecular_orbitals = molecule_info.num_molecular_orbitals
        num_spin_orbitals = 2 * num_molecular_orbitals
        num_electrons_spin_up_spin_down = (
            molecule_info.num_alpha,
            molecule_info.num_beta,
        )
        initial_state = HartreeFock(
            num_spin_orbitals, num_electrons_spin_up_spin_down, map_fermions_to_qubits
        )

        chemistry_inspired_trial = UCCSD(
            map_fermions_to_qubits,
            num_electrons_spin_up_spin_down,
            num_spin_orbitals,
            initial_state=initial_state,
        )

        trial_state = chemistry_inspired_trial
    else:
        if hardware_inspired_trial is None:
            hardware_inspired_trial = TwoLocal(
                rotation_blocks=["ry"],
                entanglement_blocks="cz",
                entanglement="linear",
                reps=2,
            )

        trial_state = hardware_inspired_trial

    # Step 3b and alternative
    if vqe:
        noise_free_quantum_environment = QuantumInstance(Aer.get_backend('statevector_simulator'))
        solver = VQE(ansatz=trial_state, quantum_instance=noise_free_quantum_environment)
    else:
        solver = NumPyMinimumEigensolver()

    # Step 4 and alternative
    ground_state = GroundStateEigensolver(map_fermions_to_qubits, solver)

    # Refinement to Step 4
    energy_surface = BOPESSampler(gss=ground_state, bootstrap=False)
    energy_surface_result = energy_surface.sample(
        fermionic_hamiltonian, perturbation_steps
    )

    return energy_surface_result
    
################################################# FUNCTION DEFINITION
def plot_energy_landscape(energy_surface_result):
    if len(energy_surface_result.points) > 1:
        plt.plot(energy_surface_result.points, energy_surface_result.energies, label="VQE Energy")
        plt.xlabel('Atomic distance Deviation(Angstrom)')
        plt.ylabel('Energy (hartree)')
        plt.legend()
        plt.show()
    else:
        print("Total Energy is: ", energy_surface_result.energies[0], "hartree")
        print("(No need to plot, only one configuration calculated.)")
    
################################################# simulation
# define molecular variation for G
molecular_variation = Molecule.absolute_stretching

#Other types of molecular variation:
#molecular_variation = Molecule.relative_stretching
#molecular_variation = Molecule.absolute_bending
#molecular_variation = Molecule.relative_bending

# define base guanine and simply arginine with one atom that is approaching guanine 
PAM_specific_molecular_variation = apply_variation_to_atom_pair(molecular_variation, atom_pair=(13, 6))
macromolecule = Molecule(geometry=
                                 [['C',  [-2.477, 5.399, 0.000]],
                                  ['N',  [-1.289,   4.551,   0.000]],
                                  ['C',  [0.023,   4.962,   0.000]],
                                  ['N',  [0.870,   3.969,   0.000]],
                                  ['C',  [0.071,   2.833,   0.000]],
                                  ['C', [0.424,   1.460,   0.000]],
                                  ['O', [1.554,   0.955,   0.000]],
                                  ['N', [-0.700,   0.641 ,  0.000]],
                                  ['C', [-1.999,   1.087,   0.000]],
                                  ['N', [-2.949,   0.139,  -0.001]],
                                  ['N', [-2.342,   2.364,   0.001]],
                                  ['C', [-1.265,   3.177,   0.000]],
                                  ['C', [-1.265,   3.177,   0.000]],
                                  ['N', [1.554,   1.955,   0.000]] # this is the nitrogen atom from arginine 
                                 ],
                                  charge=-0.5, multiplicity=1, # Guanine is slightly negative
                                  degrees_of_freedom=[PAM_specific_molecular_variation])
##

PAM_energy_surface_result = construct_hamiltonian_solve_ground_state(
   molecule=macromolecule,
   num_electrons=2,
   num_molecular_orbitals=2,
   chemistry_inspired=True,
   vqe=True,
   perturbation_steps=np.linspace(-0.5, 5, 30),
)
plot_energy_landscape(PAM_energy_surface_result)
    
