from .protocol import Paper
import math


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em;
      line-height: 1;
      display: inline-flex;
      align-items: center;
    }
    .half-star {
      display: inline-block;
      width: 0.5em;
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""


def get_empty_html():
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
      <td style="font-size: 20px; font-weight: bold; color: #333;">
          No Papers Today. Take a Rest!
      </td>
    </tr>
    </table>
    """
    return block_template


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: str = None,
    semantic_score: str = None,
    keyword_matches: str = None,
    publish_date: str = None,
):
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Published:</strong> {publish_date=publish_date}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Affiliations:</strong> {affiliations}
        </td>
    </tr>
     
    semantic_score_html = ""
    if semantic_score not in [None, "", "Unknown"]:
        semantic_score_html = f"""
        <tr>
            <td style="font-size: 14px; color: #333; padding: 8px 0;">
                <strong>Semantic Score:</strong> {semantic_score}
            </td>
        </tr>
        """

    keyword_matches_html = ""
    if keyword_matches not in [None, "", "None"]:
        keyword_matches_html = f"""
        <tr>
            <td style="font-size: 14px; color: #333; padding: 8px 0;">
                <strong>Matched Keywords:</strong> {keyword_matches}
            </td>
        </tr>
        """

    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    {semantic_score_html}
    {keyword_matches_html}
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        tldr=tldr,
        pdf_url=pdf_url,
        affiliations=affiliations,
        semantic_score_html=semantic_score_html,
        keyword_matches_html=keyword_matches_html,
    )


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8

    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high - low) / 10
        star_num = math.ceil((score - low) / interval)
        full_star_num = int(star_num / 2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">' + full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers: list[Paper]) -> str:
    parts = []

    if len(papers) == 0:
        return framework.replace('__CONTENT__', get_empty_html())

    for p in papers:
        rate = round(p.score, 1) if p.score is not None else 'Unknown'

        publish_date = getattr(p, "publish_date", None)
          if not publish_date:
              publish_date = "Unknown Publish Date"
          
        semantic_score_value = getattr(p, "semantic_score", None)
        if semantic_score_value is not None:
            semantic_score = f"{semantic_score_value:.4f}"
        else:
            semantic_score = None

        keyword_matches_value = getattr(p, "keyword_matches", None)
        if keyword_matches_value:
            if isinstance(keyword_matches_value, (list, tuple, set)):
                keyword_matches = ", ".join([str(k) for k in keyword_matches_value])
            else:
                keyword_matches = str(keyword_matches_value)
        else:
            keyword_matches = None

        author_list = [a for a in p.authors]
        num_authors = len(author_list)

        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])

        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'

        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                p.tldr,
                p.pdf_url,
                affiliations,
                semantic_score,
                keyword_matches,
                publish_date
            )
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
