# ROGII – Wellbore Geology Prediction: Zirveye Oynama Planı

> Hazırlayan: Claude · Tarih: 16 Haziran 2026
> Canlı leaderboard, forum tartışmaları ve senin verin üzerinde yapılan deneylere dayanır.
> Bu bir **yol haritasıdır** — okuyup yön onayladıktan sonra kodlamaya geçilecek.

---

## 0. Özet (önce bunu oku)

- **Şu an neredesin:** Canlı gönderim skorun **14.056**. Bu, projedeki basit **v1 CatBoost** notebook'undan geliyor. Güçlü "9.251 DWT" notebook'u Kaggle'da hata verip durmuş.
- **Zirve nerede:** #1 = **5.670**, ilk 50'nin tamamı **7.33 altında**, toplam **3123 takım**, ~**2 ay** kaldı. Public LB testin sadece **%26'sı**; final %74'lük gizli kısma göre belirlenecek.
- **En kritik bulgu (senin verinde kanıtlandı):** TVT'nin baskın sinyali **GR eşleştirmesi değil, kuyunun kendi Z (dikey) yörüngesidir.** Kuyu başına basit bir doğru — `dtvt = a·dz + b` — verinde **6.80 RMSE** veriyor. Yani 9.25'lik notebook'un asıl işi (DTW / particle filter ile GR eşleştirme) **zayıf sinyale** yatırım yapıyor.
- **Gerçekçi beklenti:** 3123 takım ve derin öğrenme cephesi varken kimse "1. olacaksın" diye söz veremez. Ama yapı şu: **14 → ~8 (geometrik), → ~7 (uzaysal + offset), → ~6 (derin MTP modeli), → 5.x (ensemble + post-process).** Her aşama ölçülebilir bir sıçrama.

---

## 1. Sahanın durumu (canlı LB, 16 Haz 2026)

| Sıra | Takım | Public skor |
|---|---|---|
| 1 | Tucker Arrants | **5.670** |
| 2 | SaintLouis | 5.785 |
| 3 | Rishikesh Jani | 6.026 |
| 4 | Ruby | 6.394 (sadece 7 gönderim) |
| 5 | Ryo Takaki | 6.429 |
| 11 | Ertuğrul Demir | 6.720 |
| 34 | Chris Deotte (GM) | 7.182 |
| 49 | zhuo wang | 7.328 |

Public notebook'lar: romantamrazov 12.6 → pilkwang 12.0 → ravaghi 9.43 → rauffauzanrambe 9.54 → **nihilisticneuralnet 9.251 (senin elindeki)**. Yani elindeki notebook **en iyi public** olan ama yine de tepeden ~3.5 RMSE geride ve **herkese açık** — tek başına asla kazandırmaz.

**Skoru not düş:** 9.25 ile 5.67 arasındaki fark, "biraz tuning" farkı değil; **yöntem** farkı.

---

## 2. Yarışma tam olarak ne (teyitli şema)

- **Görev:** Yatay kuyuda **TVT (True Vertical Thickness)** tahmini — yani jeolojik katman içindeki dikey konum. Regresyon. Metrik: **RMSE** (düşük = iyi).
- **Veri:** ~**773 train kuyusu**, ~**200 gizli test kuyusu**. Toplam 1.33 GB, 2327 dosya.
- **Her kuyu için dosyalar:**
  - `{WELL}__horizontal_well.csv`: `MD, X, Y, Z, GR, TVT_input` (her ikisinde de var) + sadece train'de: `TVT` (hedef) ve formasyon yüzeyleri `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`.
  - `{WELL}__typewell.csv`: `TVT` (dikey referans indeksi), `GR`, `Geology` (formasyon etiketi).
- **Tahmin bölgesi:** `TVT_input` topuğda (heel) bilinir, ayak (toe) kısmında **NaN** — skorlanan yer burası. NaN'in başladığı yer = **PS (Prediction Start)** noktası, sızıntı sınırı.
- **Skorlama:** Public %26 / Private %74. Host **bir aykırı kuyuyu private setten çıkardı** ve yeniden skorladı (görünür bir etkisi yok). Notebook runtime'ı bol (bir rakibin notebook'u ~4.6 saat çalışmış).
- **Kural:** Code competition. Submit = Kaggle notebook'u `submission.csv` üretir. Internet KAPALI; harici paket/ağırlık Kaggle Dataset olarak eklenmeli. Sınırlı görünür `test/` klasörü sadece örnek — gerçek test gizli ve submit'te değişiyor.

---

## 3. Neden 9.25 yetmiyor — ve tepe ne yapıyor

Forumdaki en çok oylanan teknik tartışma (Grandmaster **hengck23**, 93 yorum) + LB'nin tepesindekilerin (Tom 37., Tucker 1., sleep3r 122.) yorumları, oyunun nasıl oynandığını açık ediyor:

**A) Baskın sinyal Z yörüngesidir, GR değil.**
- Tom (37.): *"cumsum(−dz − offset), ayrık bir offset ile => **7.7 rmse**."* — neredeyse ML'siz bir geometrik formül, 9.25'lik GBM'i yeniyor.
- hengck23: *"z prior, gr prior'dan çok daha güçlü."*
- **Neden:** Test'te kuyunun PS sonrası tüm X/Y/Z yörüngesi **veriliyor**. Sondajcı formasyonu takip ederek yönlendirdiği için, yörüngedeki `dz` zaten cevabı kodluyor. Kabaca `TVT ≈ formasyon_tavanı(X,Y) − Z`.

**B) GR eşleştirmesi kötü koşullu (ill-conditioned).**
- sleep3r (122.): gerçek TVT'de bile yatay↔typewell GR korelasyonu sadece ~0.7; offset hatası birikiyor.
- Panellerde GR top-10 kapsama: normal %25.6 vs karıştırılmış %26.1 — yani GR neredeyse rastgele kadar zayıf. **Senin 9.25 notebook'un tam da buraya (DTW + 2 particle filter + beam search) yatırım yapıyor.**

**C) Hata, kuyu boyunca birikir.**
- hengck23: *"RMSE uzunlukla artıyor (hata birikimi). Post-processing yardımcı olur."*
- Kayan pencere ile çıkarım yapan modellerde her pencere bir öncekinin hatasını miras alır → öz-düzeltme şart.

**D) Cephe (~5.7–7) derin öğrenme dizi modelleri.**
- hengck23'ün yaklaşımı: **CNN/U-Net + SDF (signed distance function) + MTP (Multi-Trajectory Prediction / mixture density)** — birden çok yol hipotezi (k-beam gibi). Girdi: typewell GR × horizontal GR 2D ısı haritası + history + mask. Augmentation: flip + stretch. **Sentetik veri üretimi** (yol üret → typewell'den örnekle → gürültü ekle).
- Transformer MTP: history + x,y,z,dz ile `dtvt` yörüngesi tahmini (GR olmadan bile çalışıyor; GR'yi düzeltici olarak ekliyor).

**Senin verinde doğrulama (oracle tavanlar):**

| Yöntem (773 kuyu, drift hedefi) | RMSE |
|---|---|
| Carry-forward (drift=0) | 15.91 |
| Global `c·dz` | 15.79 (işe yaramaz) |
| **Kuyu-başı doğrusal `a·dz + b` (oracle)** | **6.80** |

Globalde `corr(drift, −dz)=0.14` (zayıf), çünkü her kuyunun **dip eğimi `a` farklı**. Ama kuyu içinde ilişki neredeyse doğrusal. Demek ki bütün iş: **her kuyu için `a` (yerel dip) ve `b` (offset) değerlerini sızıntısız olarak tahmin etmek.** Oracle tavan 6.80; tepe (5.67) bunun da altında çünkü dip'in kuyu içinde **parçalı** değiştiğini (Tom: kuyu başına ~15 kontrol noktası, ~323 satırda bir — StarSteer dip anotasyonu) ve GR ile **reset/çapa** noktalarını yakalıyorlar.

---

## 4. Kazanma mimarisi — aşamalı plan

Felsefe: **Önce baskın sinyali (Z + uzaysal jeoloji) sömür, sonra derin öğrenmeyi artık (residual) ve çoklu-hipotez için kullan.** Her aşamada CV ve LB ölç, sadece iyileştireni tut.

### Stage 0 — Çalışan hat + dürüst CV (hedef: not 14 değil, ~9.25 referans çalışsın)
- **9.25 DWT notebook'unu Kaggle'da hatasız çalıştır.** Olası sebep: `from hill_climbing import Climber` ve `ravaghi/...artifacts` gibi harici bağımlılıkların notebook'a Dataset olarak eklenmemesi; ya da `Private Test Update` sırasındaki geçici platform hatası (Ertuğrul Demir de yaşamış, "aynı notebook'u tekrar submit edince düzeldi"). **Aksiyon:** bağımlılıkları sabitle, internet-kapalı uyumlu hale getir, deterministik seed.
- **Sızıntı-güvenli CV'yi ilk işte kur:** `GroupKFold` (kuyuya göre) **+** tahmin-bölgesi farkındalığı (bir kuyuda downhole satır, daha önceki satırın tahminini beslemesin). **CV ≈ LB olduğunu kanıtla** (en az 2 farklı submit ile kalibre et). Bu olmadan her optimizasyon kör.

### Stage 1 — Geometrik Z-prior temeli (hedef: ~7.7–8.5) ⚡ en yüksek ROI
- Çekirdek: `TVT(md) = TVT_ps + (a_well · dz(md)) + b` türü, kuyu-başı eğim + offset.
- `a_well` ve `b`'yi **yalnızca yasal bilgiden** tahmin et:
  - **Heel (bilinen prefix) fit:** PS öncesi `TVT_input` vs `dz` eğiminden başlangıç `a, b`.
  - **Geometri:** `z_slope`, inklinasyon, `MD`–`Z` ilişkisi.
- Tom'un `cumsum(−dz − offset)` reçetesi (~7.7) bunun özel hali. **Bu tek başına seni 14 → ~8'e taşır ve 9.25'i geçer**, neredeyse sıfır compute ile. İlk somut zafer burada.

### Stage 2 — Uzaysal jeoloji yüzeyi (hedef: Stage 1 ile harman ~7)
- Formasyon tavanları (`ANCC = TVT + Z` vb.) **(X,Y)'nin pürüzsüz uzaysal fonksiyonu**. Komşu kuyulardan **kriging / yerel düzlem (plane) KNN** ile test kuyusunda tahmin et → `TVT = ANCC_plane(X,Y) − Z + offset`.
- hengck23 & PatrickAIForFun: ham kriging ~11 RMSE; **iyi offset ayarıyla** validation ~11 → blend'de değerli. Senin 9.25 notebook'unda zaten `FormationPlaneKNN` ve `DenseANCCImputer` var — bunları Stage 1 ile **doğru harmanlamak** kazandırır.
- `ANCC` düzlemi `TVT`'den daha doğrusal/pürüzsüz → tahmini daha kolay (hengck23'ün önerisi).

### Stage 3 — Öğrenilmiş global offset seçici (hedef: ~6.5–7)
- `cumsum(−dz)` doğru offset olmadan hatalı. **Ayrık/global offset'i seçen bir sınıflandırıcı/regresör** eğit (özellikler: heel GR↔typewell eşleşmesi, geo-plane tahmini, PS yakını `Z`).
- hengck23: *"sadece global offset'i seçecek bir classifier lazım."* Bu küçük model, Stage 1+2'nin en büyük hata kaynağını (yanlış offset) kapatır.

### Stage 4 — Derin Çoklu-Yörünge (MTP) modeli (hedef: ~6) 🎯 puanın olduğu yer
- **Girdi temsili (en kritik tasarım kararı):**
  - Dizi kanalları: `dz, x, y, z, cos/sin(heading), dtvt history, mask` (hengck23'ün transformer'ı bunları kullanıyor).
  - GR ısı haritası kanalı: `typewell_GR(tvt)` × `horizontal_GR(md)` 2D matris (CNN/U-Net için), düzeltici olarak.
- **Model:** 1D U-Net / CNN backbone → **MDN (mixture density) / MTP head** → top-K yörünge hipotezi + olasılık. Çoklu hipotez şart, çünkü çözüm birkaç mod arasında "zıplıyor".
- **Kayıp:** yerel `dtvt` kaybı **+** global `cumsum→tvt` kaybı (hengck23 önerisi). Hata birikimini cezalandırır.
- **Augmentation + sentetik:** flip, stretch; ve forward-simulation ile **sınırsız sentetik kuyu** (yol üret → typewell'den GR örnekle → gürültü ekle). Bu, GR korelasyonunu öğretmenin yolu.
- **Çıkarımda öz-düzeltme:** kayan pencere + GR çapalarıyla reset (hata birikimine karşı).

### Stage 5 — Ensemble + post-process + submit seçimi (hedef: 5.x)
- Stage 1/2/3/4 OOF tahminlerini **hill-climbing blend** (senin notebook'ta zaten Climber var).
- **Post-process:** kuyu-başı Savitzky-Golay yumuşatma, fiziksel süreklilik kısıtı, uzunlukla artan hatayı düzeltme.
- **Final submit seçimi:** Public %26 — ona **fazla güvenme**. 2 final submission'dan birini en iyi public, diğerini **en iyi CV / en sağlam (robust)** seç. Shake-up'a karşı sigorta.

---

## 5. Sızıntı-güvenli CV ve private-LB stratejisi (kazanma/kaybetme noktası)

- **GroupKFold (kuyu)** + tahmin-bölgesi farkındalığı: downhole bilgi asla geriye sızmasın.
- Her özellik **inşa gereği** sızıntısız olmalı; sınırı geçebilecek özelliği gizlemek yerine **işaretle**.
- **CV ↔ LB kalibrasyonu:** birkaç submit ile ilişkiyi doğrula; tutmuyorsa önce onu düzelt.
- Public sadece %26 → **CV'ye ve private mantığına güven.** "How much should we trust the LB?" forumda açık bir endişe. Tucker (1.) bile post-PS yörüngenin sızıntı olabileceğini tartışıyor — kuralları takip et ama CV'ni buna göre kur.

---

## 6. ~8 haftalık takvim (deadline ~ Ağustos ortası)

| Hafta | Odak | Hedef skor |
|---|---|---|
| 1 | Stage 0: notebook'u onar, sızıntı-güvenli CV, CV≈LB kalibrasyonu | çalışan 9.25 |
| 2 | Stage 1: geometrik Z-prior (kuyu-başı a,b) | ~8 |
| 3 | Stage 2: uzaysal jeoloji düzlemi + blend | ~7.3 |
| 4 | Stage 3: offset seçici classifier | ~6.8 |
| 5–6 | Stage 4: derin MTP modeli (CNN/U-Net + MDN), sentetik augmentation | ~6.2 |
| 7 | Stage 5: ensemble + post-process + öz-düzeltme | ~5.9 |
| 8 | Tuning, robustluk, 2 final submission seçimi | sağlamlaştır |

> Skorlar hedef; gerçek sıçramalar deneyle netleşir. Önemli olan: her hafta ölçülebilir, sızıntısız bir ilerleme.

---

## 7. Dürüst beklenti ve riskler

- **Garanti yok.** 3123 takım, güçlü GM'ler (Tucker Arrants, hengck23, Chris Deotte, Ertuğrul Demir) sahada. "1. olacaksın" diyen yanıltır.
- **Gerçekçi sonuç bandı:** Bu planı düzgün uygularsak **madalya bölgesine (üst %5–10) girmek** ulaşılabilir bir hedef; **podyum/1.** ise ek bir "edge" (daha iyi temsil, daha iyi sentetik üretim, daha iyi ensemble) ve biraz şans gerektirir.
- **Shake-up riski:** Public %26 yanıltıcı olabilir; sağlam CV ve dengeli final-submission seçimi bu riski yönetir.
- **Compute:** Kaggle ücretsiz GPU (~30 saat/hafta) MTP modeli için yeterli ama dar; eğitim/iterasyonu verimli kurmalıyız.

---

## 8. Hemen sonraki adımlar (kim ne yapıyor)

**Ben (Claude):**
1. Stage 0 + Stage 1'i **kodlayıp** Kaggle'a hazır notebook olarak veririm (sızıntı-güvenli CV + geometrik Z-prior). Bu tek başına 14 → ~8 hedefler.
2. Forumdaki referans notebook'ları (hengck23 `cnn-mtp-example`, `cnn-sdf-example`; ravaghi hill-climbing) senin için canlı okuyup en iyi parçaları plana entegre ederim.

**Sen (Tuna):**
1. Verdiğim notebook'u Kaggle'da çalıştırıp **public skoru + hata loglarını** bana yapıştır (iterasyon döngümüz bu). *(Not: bu seçeneği işaretlemedin — onaylarsan döngü çok hızlanır.)*
2. Onayın: hangi aşamadan başlayalım — **Stage 0 (önce 9.25'i onar)** mı, yoksa doğrudan **Stage 1 (geometrik Z-prior, en yüksek ROI)** mı?

---

## 9. Kaynaklar (forumdan, doğrulanmış)

- **Paper:** "Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning" — Alyaev — `arxiv.org/pdf/2201.01871`
- **Ders/kod:** `github.com/geosteering-no/inversion_school_geosteering`
- **hengck23 örnek notebook'ları:** `kaggle.com/code/hengck23/cnn-mtp-example`, `kaggle.com/code/hengck23/cnn-sdf-example`
- **Public baseline'lar:** ravaghi hill-climbing (9.43), nihilisticneuralnet DWT (9.251), pilkwang EDA+leakage (12.0)
- **Anahtar forum başlıkları:** "multi-trajectory prediction (MTP) with deep CNN" (hengck23), "Paradigm Shift: Why pure Tabular Models might be hitting a wall", "How Geologists Interpret Wells" (host Igor Kuvaev)
