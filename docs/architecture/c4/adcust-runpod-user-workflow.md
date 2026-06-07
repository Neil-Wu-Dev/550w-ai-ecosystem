# AdCust + RunPod 用户工作流程

| Phase | Location | User action | AdCust behavior | Important result |
|---|---|---|---|---|
| 1 | RunPod | Select an available GPU and create a pod | None | A running Linux + NVIDIA GPU server exists |
| 2 | RunPod | Read the SSH host, Direct TCP port, username and credential | None | Current SSH connection data is available |
| 3 | AdCust Compute | Click `Add Node` and enter the RunPod connection data | Persist the compute profile locally | The pod becomes a registered compute node |
| 4 | AdCust Compute | Click `Try Connection` | Perform a real SSH connection and heartbeat check | Connected or a real connection error is displayed |
| 5 | AdCust Training | Select the node and start training | Prepare runtime, download the exact model revision, upload assets, train, stream logs and download the adapter | The signed adapter is stored locally |
| 6 | RunPod | Manually stop the pod after the adapter is downloaded | AdCust only closes SSH | RunPod billing stops according to the provider's lifecycle rules |
| 7 | RunPod | Start the same pod for a later session | None | The pod may receive a new Direct TCP port |
| 8 | AdCust Compute | Open the existing node, click `Edit`, and update the port | Persist the updated connection data and reconnect | The existing node profile can be reused |
| 9A | RunPod | If no GPU is available, wait and retry later | None | The same pod may become startable later |
| 9B | RunPod | Save or use a template containing the prepared runtime, then create a replacement pod | None | A new pod is created without rebuilding the runtime manually |
| 10 | AdCust Compute | Add the replacement pod as a new node with its new SSH data | Register and verify the new compute node | Training can continue on the replacement pod |

## Core Rules

1. RunPod controls GPU availability, pod creation, start, stop, destruction, templates, and billing.
2. AdCust controls node registration, SSH verification, environment preparation, training, monitoring, and adapter downloading.
3. Stopping or disconnecting SSH in AdCust does not stop the RunPod pod.
4. A restarted RunPod pod may expose a different Direct TCP port, so the saved AdCust node must be updated before reconnecting.
5. A replacement pod has a new identity and connection configuration, so it should be registered as a new AdCust node.
