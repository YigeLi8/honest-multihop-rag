# Future work (not this build)

Deliberately out of scope for the one-week artifact:

- RL over retrieval: a self-rewarding retrieval step (tree search + policy
  optimization) that would turn the harness from measurement into improvement.
- Distributed / heterogeneous serving, and learned KV-cache eviction for
  long-context multi-hop. This build only measures single-device knobs.
- Graph learning. The graph arm here is retrieval over a static entity graph.
