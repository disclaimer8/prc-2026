# Обращение к владельцам приёмников: локальные логи за январь и июль 2026 (готово 22.09, НЕ отправлено)

**Что просим:** сырую историю приёма (tar1090/readsb `globe_history`, `chunks/`, или дампы Beast) за
**01–31.01.2026** и **01–31.07.2026** в радиусе ~10 км от **LIRF (Фьюмичино)**, **LTFM (Стамбул)** или
**LFPG (Руасси)** — то есть там, где у всех открытых сетей наземных точек ноль.

**Зачем именно они:** нужен момент, когда борт тронулся со стоянки. У нас он есть в Амстердаме и Цюрихе и
отсутствует в Риме, Стамбуле и Париже. Это не вопрос выбора архива: MLAT-ветка adsb.lol за оба месяца пуста
в этих трёх аэропортах, у Airplanes.Live дыра та же, а у FlyItalyADSB центральный архив июля потерян при
миграции систем. Наблюдение существовало только у локальных приёмников.

**Условие, без которого данные бесполезны:** правила конкурса требуют **открытой лицензии** и публикации
происхождения. Человек должен сам написать, под какой лицензией отдаёт (годятся ODbL 1.0, CC BY 4.0, CC0).
Без этой строки мы данные не берём, даже если они у него есть.

**Куда присылать:** точка приёма уже поднята (`adsbdrop` на нашем Hetzner, доступ по ключу).

---

## Текст (английский, для Discord и форумов)

> **Looking for local ADS-B receiver logs near Rome FCO, Istanbul IST or Paris CDG — January and July 2026**
>
> We are a team in the EUROCONTROL PRC Data Challenge 2026 (taxi-out prediction, open results, GPLv3 code).
> Ground-level ADS-B at LIRF, LTFM and LFPG is missing from every open archive we checked: adsb.lol (both the
> normal and the MLAT-only release), Airplanes.Live, ADSBiq, adsb.fi, adsb.one. FlyItalyADSB told us their own
> July archive was lost in a systems migration.
>
> If you ran a receiver within ~10 km of one of those airports in **January or July 2026** and still have the
> raw history (tar1090/readsb `globe_history`, `chunks/`, Beast dumps — anything with positions and timestamps),
> we would like to use it, and we can arrange the upload.
>
> One condition comes from the rules rather than from us: the data has to be under an **open licence you state
> yourself** (ODbL 1.0, CC BY 4.0 or CC0), because we must publish where every input came from. Attribution goes
> into the paper and the public repository, exactly as you write it.
>
> Happy to answer anything about what we do with it. Thanks for reading.

## Текст (итальянский, для римских групп)

> **Cerco log locali ADS-B vicino a Roma Fiumicino — gennaio e luglio 2026**
>
> Partecipiamo alla EUROCONTROL PRC Data Challenge 2026 (previsione del tempo di rullaggio; codice pubblico,
> GPLv3). I punti a terra su LIRF mancano in tutti gli archivi aperti che abbiamo verificato, e FlyItalyADSB ci
> ha confermato che il loro archivio di luglio è andato perso durante una migrazione.
>
> Se avevi un ricevitore entro ~10 km da FCO a **gennaio o luglio 2026** e conservi ancora la cronologia grezza
> (tar1090/readsb `globe_history`, `chunks/`, dump Beast), ci interesserebbe molto.
>
> Una condizione, che viene dal regolamento: i dati devono avere una **licenza aperta indicata da te**
> (ODbL 1.0, CC BY 4.0 o CC0), perché dobbiamo pubblicare l'origine di ogni dato. L'attribuzione finisce nel
> repository pubblico così come la scrivi tu.

## Текст (турецкий, для Стамбула)

> **İstanbul Havalimanı (LTFM) çevresinde yerel ADS-B kayıtları arıyoruz — Ocak ve Temmuz 2026**
>
> EUROCONTROL PRC Data Challenge 2026 yarışmasındayız (taksi süresi tahmini, kod GPLv3 ile açık). LTFM'de yer
> seviyesindeki ADS-B noktaları kontrol ettiğimiz tüm açık arşivlerde yok.
>
> **Ocak veya Temmuz 2026**'da havalimanına ~10 km mesafede bir alıcı çalıştırdıysanız ve ham geçmiş kayıtlar
> (tar1090/readsb `globe_history`, `chunks/`, Beast dökümleri) hâlâ duruyorsa bizimle paylaşır mısınız?
>
> Tek şart yarışma kurallarından geliyor: veriler **sizin belirteceğiniz açık bir lisansla** olmalı (ODbL 1.0,
> CC BY 4.0 veya CC0), çünkü her verinin kaynağını yayımlamak zorundayız.

---

## Куда писать (по убыванию шанса)

| Адресат | Почему он | Как |
|---|---|---|
| Discord adsb.lol (#feeders) | их фидеры и есть источник нашего архива; локальные логи у людей живут дольше, чем в сети | пост в канале фидеров |
| Discord Airplanes.Live | вторая сеть с теми же людьми, у них есть карта фидеров | пост |
| FlyItalyADSB, `info@flyitalyadsb.com` | уже отвечали; спросить не архив, а **контакт римских фидеров** | письмо |
| r/ADSB на Reddit | англоязычный текст выше | пост |
| Форум ADSBexchange / их Discord | их бесплатный образец показывает 52 борта на земле в LIRF 01.07.2026 — значит, приёмники там были именно у их фидеров | пост |
| Турецкие споттерские группы (Facebook, Telegram) | Стамбул | пост на турецком |

**Что НЕ делать:** не просить выгрузку у самих сетей (проверено, архивов нет) и не принимать данные без
явной лицензии — иначе приз аннулируется.
