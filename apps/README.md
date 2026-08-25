# Applications

This directory owns runnable FlashDreams v2 applications and demos. Application
packages may provide UI loops, sessions, client-facing argument parsing, sample
assets, and application-level tests.

Model implementation and configuration adapters belong under
`integrations_v2/<integration>/impl/` and
`integrations_v2/<integration>/apps/<slug>/`, respectively.
An adapter may select a pipeline config and provide narrowly scoped hooks to a
shared app, but it must not duplicate an application, session, model loop, or UI
loop for one model.

The shared text-to-video implementation is in [`t2v`](./t2v/README.md).
