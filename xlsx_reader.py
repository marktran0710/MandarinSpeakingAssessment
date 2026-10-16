"""Minimal read-only xlsx reader (stdlib only, no openpyxl): sheet name -> list of rows (lists of str)."""
import re, sys, zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters: n = n * 26 + ord(ch) - 64
    return n - 1

def read_xlsx(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in rels}
        sheets = {}
        for sh in wb.find("m:sheets", NS):
            rid = sh.get("{%s}id" % NS["r"])
            part = target[rid].lstrip("/")
            part = part if part.startswith("xl/") else "xl/" + part
            root = ET.fromstring(z.read(part))
            rows = []
            for row in root.iter("{%s}row" % NS["m"]):
                cells = {}
                for c in row.findall("m:c", NS):
                    v = c.find("m:v", NS)
                    t = c.get("t")
                    if t == "inlineStr":
                        text = "".join(x.text or "" for x in c.iter("{%s}t" % NS["m"]))
                    elif v is None: text = ""
                    elif t == "s": text = shared[int(v.text)]
                    else: text = v.text
                    cells[col_index(c.get("r"))] = text
                if cells:
                    rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
            sheets[sh.get("name")] = rows
        return sheets
