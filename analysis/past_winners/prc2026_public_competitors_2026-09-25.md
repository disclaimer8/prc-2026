# Публичные репозитории конкурентов PRC 2026 — обход 25.09 (не по индексу тем)

Вопрос Denys: репозитории могли не попасть в индекс по темам GitHub — искать иначе. Сделано:
1. **Поиск по коду** GitHub по уникальным строкам датасета: `TAXITIME_SEC_mvt`, `AOBT_3_flt`, `MVT_ID_mvt`, `LOBT_flt`,
   `submitting.parquet`, `prc-2026-datasets` (code search видит содержимое файлов, а не только описания).
2. **Поиск репозиториев по имени**: `prc-2026`, `prc2026`, `prc-challenge`, `taxi-out`, `taxiout`, `data-challenge taxi`,
   `opensky taxi 2026`, только с пушами после 01.08.2026.
3. GitLab, Hugging Face (модели и датасеты), Codeberg по тем же словам; веб-поиск по «PRC Data Challenge 2026».
4. Проверка: наш `disclaimer8/prc-2026` в результатах code search — потому что поиск авторизован и видит **свои** приватные
   репозитории; `gh repo view` подтверждает `isPrivate: true`.

## Доска на 25.09, 19:10 (лучший файл команды, 192 команды)

| # | команда | RMSE | репозиторий |
|---|---|---:|---|
| 1 | jolly-lobster | 225,8391 | не найден |
| 2 | vigorous-whistle | 230,8057 (v27, новый) | не найден |
| 3 | quick-boat | 233,1847 | не найден |
| 4 | zesty-puzzle | 235,8294 | не найден |
| 5 | gentle-tractor | 235,9190 | не найден |
| 6 | jovial-uniform | 238,5791 | не найден (Piyush Patil в Discord упоминал `jovial-uniform_v1`) |
| **7** | **gentle-igloo (мы)** | **239,9169** | приватный до 09.10 |
| 8 | upstanding-firefly | 241,1464 | не найден |
| 9 | enthusiastic-daisy | 243,0108 (v765!) | не найден |
| 10 | youthful-giraffe | 244,1236 | не найден |

**До третьего места 3,27 с** (было 5,77 утром: quick-boat не улучшился, а vigorous-whistle обошёл его с 234,14 до 230,81).

## Найденные репозитории (все ниже нас; новые относительно списков 15–16.09 помечены ★)

| репозиторий | команда | заявленный RMSE | что внутри |
|---|---|---:|---|
| ★ skylinkapi/prc-data-challenge-2026-kind-mango | kind-mango | 281,87 (v67) | README 33 КБ; сознательно **не** используют `AOBT_3_flt` на ranking («exploiting the ranking process» — фразы нет в правилах, а 08/16.09 организатор прямо разрешил) |
| ★ radekacar/joyous-rainbow | joyous-rainbow | 299,23 (91/191) | разбор ошибки по группам |
| ★ sergiuv11/prc-data-challenge-2026 | jubilant-vase | 286,656 | без внешних данных; METAR пробовали |
| ★ satam2/knowledgeable-helicopter | knowledgeable-helicopter | локально 331/266 | CatBoost, пайплайн |
| ★ AbdullahKhan-x/PRC-2026 | logical-ladder | — | заготовка |
| ★ fenil264/prc-2026-taxi-out | joyful-lemon | — | заготовка |
| ★ ahmetabdullahgultekin/prc-taxiout-2026 | ? (ждали доступа) | — | реализация референса PRU (P10 по стоянка×ВПП, у нас E9 — отклонён), METAR IEM, OurAirports |
| Phoenix-Ops-LTD/prc2026-taxiout | zestful-fountain | 273,56 (v12, 12/131 на 13.09) | CatBoost на остатке от NM-времён; «clock overlay»: при `AOBT_3 − LOBT > 7200` заменять прогноз на MVT − LOBT (31 строка, −0,57 локально) — родственно нашему окну LOBT |
| javidmardanov/PRC-Data-Challenge-2026 | (elegant-alligator) | 268,56 (v6, 14/147) | TimesFM-3 + CatBoost, смесь экспертов по пропускам |
| ChandanHegde07/OpenAir | likable-eagle | 288,90 (v8) | журнал E1–E70: OPDI/OpenSky для непокрытых AOBT — REJECT («first_seen = lift-off», парковочных событий FCO 0) |
| victoralcadi/prc-data-challenge-2026 | victoralcadi | — | CLI-обвязка |
| Necklacesaura, Chieler, Enzoferrariz, georgejhanlon, RavindraTarunokusumo, ruchir321, kagervig, limxj1985 | — | — | пустые или заготовки |
| LeeMarshall1113/prc-taxiout-2026 (jolly-lobster? — так значилось 15.09) | — | — | **репозиторий удалён/скрыт (404)** |

GitLab, Hugging Face, Codeberg: ничего по конкурсу. Kaggle: ничего.

## Вывод

Ни одна из шести команд выше нас не публикует код (как и мы). Единственная новая мысль из найденного — «clock overlay»
zestful-fountain (замена прогноза на MVT − LOBT при разрыве AOBT−LOBT > 2 ч), но у нас окно LOBT уже жёстко клипует
прогноз, а край окна поднимается по E36/№36; сама по себе идея у них дала −0,57 локально на 31 строке — того же
масштаба, что наш подъём у края (−0,139 CV). Репозиторий, который 15.09 числился за jolly-lobster, исчез.
