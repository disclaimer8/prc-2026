# Local ADS-B receivers and communities near FCO / CDG / IST (research 2026-09-26)

What I checked: web search, Discord invite API (member and online counts as of today), the adsb.lol MLAT sync map (`mlat.adsb.lol/api/0/mlat-server/0A/sync.json`), FlightAware Discourse search and GitHub search. Nothing was posted anywhere.
adsb.fi and airplanes.live are behind Cloudflare, so I could not read their feeder or MLAT maps. Reddit's API blocked me, so there are no subreddit sizes here.

**Main finding:** none of the three cities has a dedicated local ADS-B Discord or Telegram group that I could find. The most direct way in is through named feeders on the adsb.lol MLAT map, contacted via adsb.lol.
Caveat on ground coverage: every feeder found is 10–40 km from its airport. Locations on the MLAT map are deliberately blurred ("approximate" on the map's own page). Receivers that far out will rarely decode aircraft on stands. Only the first entry below has a realistic chance of seeing taxiways.

## Ranked list

1. **adsb.lol MLAT-map feeders near the airports** (named stations). Map: https://mlat.adsb.lol/syncmap/ ; per-feeder page: `https://mlat.adsb.lol/synctable/feeder.html?0A&<name>`
   - FCO:
     - `T-LIRF110` at about 41.71,12.27, roughly 10 km SE (Ostia/Acilia side). Best candidate. The name looks like a Flightradar24 radar ID (T-LIRF###), so the owner also feeds FR24.
     - `mrchrhd-rm` about 17 km (same owner also runs `mrchrhd-pg`).
     - `mabio-adsb-antenna` 21 km; `Rader-Roma` 21 km; `26634` 24 km (15 MLAT peers, the best synced); `geekice` 23 km (possibly github.com/GeekIce, not confirmed).
   - CDG:
     - `T-LFPG3` about 13.5 km (FR24-style ID).
     - `LILLE-ANTENNA` 14 km; `Montmorency` 20 km; `LFPO198` 20 km; `max` 23 km; `noop-adsb` 23 km; `Le_Plessis_Trevise` 24 km.
   - IST: only `lostsailor` about 37 km and `Hasanpasa` about 41 km (Kadıköy), both with 0 MLAT peers.
   - Contact: adsb.lol exposes no feeder contact details. Post on adsb.lol Zulip (https://adsblol.zulipchat.com, which its docs list as the main channel), IRC #adsblol on Libera, or GitHub (github.com/adsblol). Ask the maintainers to forward the request to the named stations.
   - Language: English.
2. **Airplanes.Live Discord**: https://discord.com/invite/adsb (6,836 members, 909 online). English. Many EU feeders are members. Ask in a feeder or general channel, and read the rules channel first. This fits with your 18.09 email.
3. **Official ADS-B Exchange Group Discord**: https://discord.com/invite/official-ads-b-exchange-group-1141047768266322061 (13,205 members, 1,336 online). The largest feeder population. Its members run their own readsb and globe_history, and many feed several aggregators at once. Post a "looking for local receiver logs near LIRF/LFPG/LTFM, Jan/Jul 2026" message in the community channel, not in support.
4. **Ground Control / TheAirTraffic Discord**: https://discord.gg/pSJk5HG (10,547 members, 705 online). TheAirTraffic (globe.theairtraffic.com) is not on your known list. It says it offers "real-time and historical" data, and adsb.lol's globe_history already merges its feed. Ask the admins directly whether they have raw data or ground positions for these airports. They have not published a licence.
5. **SDR Enthusiasts Discord** (docker ultrafeeder, tar1090 and readsb authors): https://discord.gg/sTf9uYF (2,741 members, 412 online). English. These operators are the most likely to keep `globe_history` or `tar1090_archive` locally. Use the off-topic or showcase channel.
6. **ADS-B / SDR Feeder Image (adsb.im) Discord**: https://discord.com/invite/HjaEg53VvN (498 members, 72 online). adsb.im images keep local history by default. English.
7. **ADSBItalia**: a new Italian network (https://www.adsbitalia.it/). Discord https://discord.gg/2wFZgCVHYx (21 members); Telegram https://t.me/adsbitalia (5 subscribers) and https://t.me/adsbitalia_community (7 members). Italian. Very small, but the admins are Italian feeders and a direct DM is cheap.
8. **FlyItalyADSB** (already known; July 2026 is lost). It may still hold **January 2026**. Contact: X @FlyItalyAdsb, Facebook /flyitalyadsb, github.com/flyitalyadsb. Its site returns 403 to scripts, so I could not read its community links.
9. **Roma Spotters Club** (founded 2009, about 60 members, based at FCO; spotters, not a receiver network): FB https://www.facebook.com/Romaspottersclub/ , X @SpottersRoma , IG @romaspotters. Italian. Some members are airport staff. Ask whether anyone runs a receiver at Fiumicino, Isola Sacra or Focene.
10. **Roissy-CDG LFPG Spotters Group** (Facebook): https://www.facebook.com/groups/631474594958975/ ; also the page CDG Spotter https://www.facebook.com/CDGSpotter/ (about 18k likes). French. Same kind of ask.
11. **SDR (Yazılım Tabanlı Radyo) RTL-SDR Türkiye** (Facebook group): https://www.facebook.com/groups/sdrturkiye/ . Turkish. This is the only Turkish SDR community I found. Istanbul-area ADS-B hobbyists would most likely be here. Also relevant: iGA Spotter (IG @igaspotter; the official spotter terrace faces runway 1) and TRAC Kadıköy branch https://trackadikoy.org.tr/ (the radio-amateur club in the same area as the `Hasanpasa` feeder).
12. **French general SDR and radio communities** (lower yield):
    - Radio Discord France: https://discord.com/invite/radio-discord-france-834352584605040653 (607 members).
    - Forum rtl-sdr France: https://rtl-sdr.forumactif.com/
    - framboise314 ADS-B thread: https://forums.framboise314.fr/viewtopic.php?f=36&t=65
    - TetraHub SDR section: https://forum.tetrahub.net/
13. **Italian radio-amateur forums** (lower yield):
    - https://www.forumradioamatori.it/
    - forum.fibra.click thread 44830 on an ADS-B antenna in Rome with a view of FCO and CIA: https://forum.fibra.click/d/44830-perdita-efficienza-antenna-ads-b (403 to scripts; the operator's name was not read).
    - Digiland Telegram Group Italia: https://digilanditalia.wordpress.com/

## Existing public archive for Jan or Jul 2026 near these airports
None found apart from the one you already use: adsblol/globe_history_2026, which includes the FlyItalyADSB and TheAirTraffic feeds.
- The FlightAware thread on `tar1090_archive` daily outlines (discussions.flightaware.com/t/100613) is about range outlines, not traces.
- The GitHub repo yyigitturan/ist-airport-decision-support-system uses an LTFM ADS-B archive, but the README says it follows OpenSky conventions. Its source is probably OpenSky. A contact address is on the maintainer's GitHub profile.

## How to word the request
- Say exactly what you need: ICAO, dates (01–31 Jan and 01–31 Jul 2026), and any of: globe_history, readsb `--write-json` history, or SBS/Beast logs.
- Say what it is for (a research competition), how it will be credited, and that you will not redistribute it.
- Post once per server, in the general or community channel. Do not DM members you don't know until an admin agrees.
