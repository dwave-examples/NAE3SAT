import os
from utilities import nae3sat
import minorminer
import dimod
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.preprocessing import ScaleComposite
import matplotlib.pyplot as plt


# Define problems
problem_size = 75
seed = 42
rho_list = [2.1, 3.0]

# Create directory for plots
if not os.path.exists("./plots/"):
    os.makedirs("./plots/")


# Get an Advantage sampler
adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"))

# Get an Advantage2 prototype sampler
adv2p_sampler = DWaveSampler(solver=dict(topology__type="zephyr"), profile="adv2")


# Generate two NAE3SAT problems with clause-to-variable ratio rho 2.1 and 3.0
for rho in rho_list:
    print(f"making nae3sat problem with rho={rho} and N={problem_size}")
    bqm = nae3sat(num_variables=problem_size, rho=rho, seed=seed)
    for sampler_name, sampler in ("Adv", adv_sampler), ("Adv2_proto", adv2p_sampler):

        # Find minor embedding
        print(f"minor embedding problem into {sampler_name}")
        embedding = minorminer.find_embedding(
            dimod.to_networkx_graph(bqm), sampler.to_networkx_graph()
        )

        # Plot chain length distributions
        plt.figure(rho * 100)
        plt.hist(
            [len(chain) for q, chain in embedding.items()],
            label=sampler_name,
            alpha=0.7,
        )
        plt.xlabel("Embedding chain length")
        plt.title(f"$\\rho={rho}$, $N={problem_size}$")
        plt.legend()
        plt.savefig("./plots/rho_{}_chain_length.png".format(int(rho * 100)))

        # Solve problem
        print(f"sending problem to {sampler_name}")
        sampleset = FixedEmbeddingComposite(
            ScaleComposite(sampler),
            embedding=embedding,
        ).sample(
            bqm,
            quadratic_range=sampler.properties["extended_j_range"],
            bias_range=sampler.properties["h_range"],
            chain_strength=3,
            num_reads=100,
        )

        # Plot energy distributions
        energies = [
            e
            for e, o in zip(sampleset.record.energy, sampleset.record.num_occurrences)
            for _ in range(o)
        ]
        plt.figure(rho * 100 + 1)
        plt.hist(energies, label=sampler_name, alpha=0.7)
        plt.xlabel("Energy")
        plt.title(f"$\\rho={rho}$, $N={problem_size}$")
        plt.legend()
        plt.savefig("./plots/rho_{}_energies.png".format(int(rho * 100)))
