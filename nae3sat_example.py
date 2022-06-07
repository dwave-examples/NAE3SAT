# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

import dimod
import matplotlib.pyplot as plt
import minorminer

from dimod.generators import random_nae3sat
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.preprocessing import ScaleComposite


# Define problems
num_variables = 75
rho_list = [2.1, 3.0]  # the clause-to-variable ratio

# Create directory for plots
os.makedirs(os.path.join(os.path.dirname(__file__), "plots"), exist_ok=True)

# Get an Advantage sampler
adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"))

# Get an Advantage2 prototype sampler
adv2p_sampler = DWaveSampler(solver=dict(topology__type="zephyr"))

# Generate two NAE3SAT problems with clause-to-variable ratio rho 2.1 and 3.0
for rho in rho_list:
    print(f"\nCreating an NAE3SAT problem with rho={rho} and N={num_variables}")

    num_clauses = round(num_variables * rho)

    bqm = random_nae3sat(num_variables, num_clauses, seed=42)

    for sampler in (adv_sampler, adv2p_sampler):

        # Find minor embedding
        print(f"\nMinor embedding problem into {sampler.solver.name}")
        embedding = minorminer.find_embedding(
            dimod.to_networkx_graph(bqm), sampler.to_networkx_graph()
        )

        # Plot chain length distributions
        chain_lengths = [len(chain) for q, chain in embedding.items()]
        plt.figure(rho * 100)
        plt.hist(
            chain_lengths,
            label=sampler.solver.name,
            alpha=0.7,
            bins=max(chain_lengths) - min(chain_lengths),
        )
        plt.xlabel("Embedding Chain Length")
        plt.ylabel("Count")
        plt.title(f"$\\rho={rho}$, $N={num_variables}$")
        plt.legend()
        plt.savefig(f"./plots/rho_{int(rho * 100)}_chain_length.png")

        # Solve problem
        print(f"Sending problem to {sampler.solver.name}")
        sampleset = FixedEmbeddingComposite(
            ScaleComposite(sampler),
            embedding=embedding,
        ).sample(
            bqm,
            quadratic_range=sampler.properties["extended_j_range"],
            bias_range=sampler.properties["h_range"],
            chain_strength=3,
            num_reads=100,
            auto_scale=False,
            label='Example - NAE3SAT',
        )

        # Plot energy distributions
        plt.figure(rho * 100 + 1)
        plt.hist(
            sampleset.record.energy,
            weights=sampleset.record.num_occurrences,
            label=sampler.solver.name,
            alpha=0.7,
        )
        plt.xlabel("Energy")
        plt.ylabel("Count")
        plt.title(f"$\\rho={rho}$, $N={num_variables}$")
        plt.legend()
        plt.savefig("./plots/rho_{}_energies.png".format(int(rho * 100)))

print("\nResults saved under the plots folder.\n")
