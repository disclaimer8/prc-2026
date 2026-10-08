# Organiser clarifications used in this solution

The challenge organiser (EUROCONTROL PRC, user `espinielli`) answered participants'
questions in the public `#prc-data-competition` channel of the OpenSky Community
Discord server. The statements below are the ones our design relies on. They are
quoted verbatim with the UTC time of the message, so the organiser can verify them.
Questions are paraphrased, and other participants are not named.

## What the model may use from the challenge data

| Date and time (UTC) | Question (paraphrased) | Organiser's answer (verbatim) | How we use it |
|---|---|---|---|
| 2026-09-09 19:40 | Is feature engineering on the provided data allowed? | "feature engineering is of course permitted, do whatever makes sense to you with the provided data. Your model could take into account what is occupied, what is coming, ..." | Neighbour, queue and stand-occupancy features from other movements |
| 2026-09-16 10:08 | Which ranking columns may the model use? | "BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are masked because that's linked to what you have to predict. The rest in the same row and other rows is available to your model." | NM timestamps of the same flight, including `AOBT_3_flt`, and the take-off time |
| 2026-09-19 07:56 | Is the model meant for tactical use? | "the model is for post-ops, not for tactical use." | The model is a retrospective reconstruction, as stated in the README |

## External data

| Date and time (UTC) | Question (paraphrased) | Organiser's answer (verbatim) | How we use it |
|---|---|---|---|
| 2026-09-03 10:00 | May OpenSky Network trajectories be used? | "... as for now, OSN state vector is not to be considered an external source of open data." | We use no OpenSky state vectors or Trino data |
| 2026-09-06 15:21 | Can OPDI be used? | "OPDI milestones for now do not cover off-block, but you can eventually use it if you find a purpose for the challenge." | OPDI events at LSZH (final correction stage) |
| 2026-09-17 12:19 | Two sources with conditions: EUROCONTROL's daily airport files (ATFM slot adherence, arrival ATFM delay, airport traffic) under a non-commercial disclaimer, and the adsb.lol historical archive under ODbL 1.0. Are they acceptable if declared? | "can be considered open data for the challenge...I am curious what can be used from these" | Destination arrival ATFM delay (E163); adsb.lol ground tracks (all surface stages and the trajectory network) |
| 2026-09-19 07:59 | Are there restrictions on using trajectory data to derive off-block times? | "there are no such restrictions: if you can find open trajectory data with good ground coverage and extract off-block times, we are ok. Practically speaking it won't be possible." | Own-aircraft surface stages |
| 2026-09-26 12:52 | May adsb.lol (ODbL), including the flight's own ground positions after pushback, be used? | "As said before open data sources can be usable to devise a better model." | As above |
| 2026-09-27 07:03 | Where do the `_mvt` columns come from; may dashboard figures be used? | "yes _mvt is from APDF, if you need to cite dashboard figures, do it. They are publicly available." | Context for the ansperformance.eu aggregates |
| 2026-10-01 14:39 | May the flight's own post-pushback ADS-B track (first movement) be a feature for that flight? | "the rule about open data applies: anything you think that helps, use it!" | Own-aircraft surface stages |
| 2026-10-01 14:41 | (general) | "If no open license is present it is not open" | Applied to every source in [SOURCES.md](../SOURCES.md) |
| 2026-10-04 03:51 | Is OPDI, published without a licence, open for eligibility? | "yes open" | OPDI events at LSZH |

## Evaluation and prize review

| Date and time (UTC) | Organiser's statement (verbatim) |
|---|---|
| 2026-09-24 16:17 | "... the goal of the challenge is on designing a MODEL, its documentation, its implementation...the ultimate aim is to then share ideas/implementations and maybe in the future apply them to other airports or to identify ways to improve..." |
| 2026-10-01 14:43 | "Just a reminder for all: we will use the latest commit before the deadline for your repo. So prepare in time: if you document after the deadline, sorry it doesn't count." |
| 2026-10-02 15:21 | "but a big part of the final ranking will go to the model design and documentation...ranking is just a part of it." |
| 2026-10-03 13:55 | "The rules are as I stated: no final round, review (AI and human!) of the top ranked submissions to assign prizes." |

The full channel history is public to members of the OpenSky Community Discord
server. The team's local exports used to compile this page are not part of the
repository.
