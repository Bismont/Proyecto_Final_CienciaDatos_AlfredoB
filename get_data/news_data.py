import requests
import pandas as pd
from datetime import timedelta

def fetch_gdelt_single_domain(date, domain, max_records=250):
    date_str = date.strftime("%Y%m%d")

    # SIN paréntesis para un solo dominio
    q = f"domain:{domain}"

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={q}"
        f"&mode=artlist"
        f"&maxrecords={max_records}"
        f"&format=json"
        f"&startdatetime={date_str}000000"
        f"&enddatetime={date_str}235959"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        data = r.json()
        arts = data.get("articles", [])
        rows = []

        for a in arts:
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "title": a.get("title", ""),
                "domain": a.get("domain", ""),
                "url": a.get("url", ""),
                "language": a.get("language", ""),
                "source": "gdelt"
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"ERROR en dominio {domain}: {e}")
        return pd.DataFrame([])

def fetch_gdelt_all_domains_day(date, domains, max_records=250):
    dfs = []
    print(f"Día {date.strftime('%Y-%m-%d')}")
    
    for dom in domains:
        print(f"    → Bajando de {dom} ...", end="")
        df_dom = fetch_gdelt_single_domain(date, dom, max_records=max_records)
        print(f" {len(df_dom)}")
        df_dom['date'] = date.strftime('%Y-%m-%d')
        df_dom['fetched_art'] = len(df_dom)
        dfs.append(df_dom)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame([])

def fetch_gdelt_range_by_domain(start_date, end_date, domains, max_records=250):
    all_rows = []
    current = start_date

    while current <= end_date:
        df_day = fetch_gdelt_all_domains_day(current,
                                             domains,
                                             max_records=max_records)
        all_rows.append(df_day)
        current += timedelta(days=1)

    return pd.concat(all_rows, ignore_index=True)

if __name__ == '__main__':
    DOMINIOS_MX = [
        'proceso.com.mx', 'eluniversal.com.mx', 'milenio.com',
        'excelsior.com.mx', 'aristeguinoticias.com', 'elimparcial.com',
        'jornada.com.mx', 'animalpolitico.com', 'eleconomista.com.mx',
        'elfinanciero.com.mx', 'expansion.mx',
    ]

    from datetime import datetime

    start = datetime(2025, 1, 1)
    end   = datetime(2025, 11, 19)   # prueba corta

    df_raw_2025 = fetch_gdelt_range_by_domain(start,
                                              end,
                                              DOMINIOS_MX,
                                              max_records=250)

    print("TOTAL TITULARES 2025:", len(df_raw_2025))
    df_raw_2025.to_csv('datos_2025.csv', index=False)

    start = datetime(2024, 1, 1)
    end   = datetime(2024, 12, 31)   # prueba corta

    df_raw_2024 = fetch_gdelt_range_by_domain(start, end, DOMINIOS_MX, max_records=250)

    print("TOTAL TITULARES 2024:", len(df_raw_2024))
    df_raw_2024.to_csv('datos_2024.csv', index=False)

        