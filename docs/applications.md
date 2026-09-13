# Applications

Keystone is a rental application service. Tenants submit one application and share it with any landlord that uses Keystone.

## Statuses

An application moves through these statuses: `submitted`, `screening`, `approved`, `declined`, `expired`.

- `submitted`: we received it and are running identity verification.
- `screening`: the landlord is reviewing it. Most landlords respond within 3 business days.
- `approved` / `declined`: the landlord's decision. We email you either way.
- `expired`: applications expire 30 days after submission if the landlord does not respond.

## Checking status

Every application has an id in the form `APP-` followed by digits (for example `APP-1042`). You can check status in the app under Applications, or ask support with your application id. Support can only look up applications by exact id.

## Editing an application

You can edit an application while it is `submitted`. Once it moves to `screening` the landlord has a snapshot and edits create a new application.
