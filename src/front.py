def render_html(tokens, labels):
    """
    Tô màu highlight theo nhãn IOB, với màu khác nhau cho PER, ORG, LOC
    """
    label_colors = {
        "PER": "lightcoral",   # đỏ nhạt
        "ORG": "lightblue",    # xanh nhạt
        "LOC": "lightgreen",   # xanh lá nhạt
    }

    html = ""
    current_label = None

    for tok, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_label:
                html += "</span> "
            current_label = label[2:]
            color = label_colors.get(current_label, "lightgray")
            html += f"<span style='background-color:{color};padding:2px;border-radius:4px;' title='{current_label}'>{tok}"
        elif label.startswith("I-") and current_label:
            html += f" {tok}"
        else:
            if current_label:
                html += "</span> "
                current_label = None
            html += f"{tok} "

    if current_label:
        html += "</span>"

    return f"<div style='font-family:monospace;font-size:16px'>{html.strip()}</div>"
