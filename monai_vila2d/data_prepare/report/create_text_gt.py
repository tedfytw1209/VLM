#!/usr/bin/env python

import copy
import jieba
import json
import multiprocessing
import numpy as np
import os
import random
import re
import spacy
import section_parser as sp
import sys
import warnings

from pathlib import Path


random.seed(10)


def remove_newline(text):
    return text.replace("\n", "").replace("\r", "").strip()


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(paragraph):
    # Regular expression to split sentences but avoid splitting on "a.m.", "p.m.", and "Dr."
    # The period following a.m., p.m., and Dr. is not treated as sentence-ending punctuation
    sentence_endings = re.compile(r"(?<!a\.m)(?<!p\.m)(?<!Dr)\.(?!\d)|(?<=[!?])")

    # Split based on the regex
    sentences = sentence_endings.split(paragraph)

    # Add the punctuation back to the sentences
    refined_sentences = []
    for i in range(len(sentences) - 1):
        sentence = sentences[i].strip()
        if sentence:
            # Add the appropriate punctuation back
            refined_sentences.append(
                sentence
                + paragraph[
                    paragraph.find(sentences[i])
                    + len(sentences[i]) : paragraph.find(sentences[i])
                    + len(sentences[i])
                    + 1
                ]
            )

    # Handle the last sentence if it was split correctly
    if sentences[-1].strip():
        refined_sentences.append(sentences[-1].strip())

    return refined_sentences


def remove_sentences_with_underscore(sentences):
    # Filter out sentences that contain the underscore character
    return [sentence for sentence in sentences if "_" not in sentence]


def refine_numbered_sentences(sentences):
    # Regular expression to match "1. ", "2. ", etc.
    pattern = re.compile(r"^\d+\.\s")

    refined_sentences = []
    for sentence in sentences:
        # Check if the sentence is just "1." or "2." etc.
        if re.fullmatch(r"\d+\.", sentence.strip()):
            continue  # Skip sentences that only contain "1." or "2." etc.
        # Refine the sentence by removing the number followed by ". "
        refined_sentences.append(pattern.sub("", sentence))

    return refined_sentences


def skip_to_first_letter(sentences):
    refined_sentences = []

    for sentence in sentences:
        # Find the position of the first letter
        for i, char in enumerate(sentence):
            if char.isalpha():
                # Capitalize the first letter and append the refined sentence
                refined_sentences.append(sentence[i].upper() + sentence[i + 1 :])
                break
        else:
            # If no letter is found in the sentence, keep it as is
            refined_sentences.append(sentence)

    return refined_sentences


def capitalize_first_letter(sentences):
    refined_sentences = []

    for sentence in sentences:
        # Check if the first character is a letter
        if sentence and sentence[0].isalpha():
            # Capitalize the first letter
            refined_sentences.append(sentence[0].upper() + sentence[1:])
        else:
            # Raise an error if the first character is not a letter
            print(sentences)
            raise ValueError(
                f"First character is not a letter in sentence: '{sentence}'"
            )

    return refined_sentences


def remove_before_colon(sentences):
    # Replace only when a colon is followed by a space
    refined_sentences = [
        sentence.split(": ", 1)[-1] if ": " in sentence else sentence
        for sentence in sentences
    ]

    return refined_sentences


def replace_abbreviations(text):
    # Replace specific abbreviations
    pattern = re.compile(r"a\.m\.\s([A-Z])")
    text = pattern.sub(r"am_. \1", text)
    pattern = re.compile(r"p\.m\.\s([A-Z])")
    text = pattern.sub(r"pm_. \1", text)

    text = text.replace("a.m.", "am_")
    text = text.replace("p.m.", "pm_")
    text = text.replace("Dr.", "Dr_")
    text = text.replace(" VIEW ", " VIEW_. ")
    text = text.replace("_: ", "_. ")

    return text


def add_period_if_missing(sentences):
    # Regular expression to check if the sentence ends with a punctuation mark
    pattern = re.compile(r"[.!?]$")

    refined_sentences = []
    for sentence in sentences:
        # If the sentence doesn't end with punctuation, add a period
        if not pattern.search(sentence.strip()):
            refined_sentences.append(sentence.strip() + ".")
        else:
            refined_sentences.append(sentence.strip())

    return refined_sentences


def partition_into_paragraphs(report):
    # Split the report based on double newlines (i.e., paragraphs separated by empty lines)
    paragraphs = report.split("\n\n")

    # Remove leading/trailing spaces from each paragraph and exclude empty paragraphs
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    return paragraphs


def remove_duplicate_sentences(sentences):
    seen = set()  # To keep track of sentences we have already seen
    unique_sentences = []

    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(
                sentence
            )  # Add sentence to the result if not seen before
            seen.add(sentence)  # Mark this sentence as seen

    return unique_sentences


def remove_single_char_sentences(sentence_list):
    """
    This function removes sentences with only one character from the list.
    """
    return [sentence for sentence in sentence_list if len(sentence) > 6]


def find_duplicates(input_list):
    """
    This function returns a list of duplicated values in the input list.
    """
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


def replace_appearance(text, target_word, replacement_word, nth_app):
    """
    Replace the second appearance of a target_word in the text with a replacement_word.
    """
    # Split the text at each occurrence of the target_word
    parts = text.split(target_word)

    if len(parts) <= 2:
        # If the target_word appears less than 2 times, no second replacement is needed
        return text

    # Rebuild the string: join the parts, but replace the second occurrence
    modified_text = (
        target_word.join(parts[:nth_app])
        + replacement_word
        + target_word.join(parts[nth_app:])
    )

    return modified_text


def remove_question_sentences(sentences):
    """
    Remove sentences that contain question marks from the list.
    """
    return [sentence for sentence in sentences if '?' not in sentence]


# for mimic-cxr-jpg training set
def customize_report(report, text_filepath):
    splits = report.split("FINAL REPORT")
    if len(splits) > 1:
        report = splits[1]
        for _i in range(2, len(splits)):
            if len(splits[_i]) > len(report):
                report = splits[_i]
        report = "                                 FINAL REPORT\n" + report

    key_pool = [
        "CHEST, TWO VIEWS:",
        "CHEST, TWO VIEWS.",
        "CHEST TWO VIEWS:",
        "PORTABLE CHEST:",
        "CHEST AND PELVIS FILMS:",
        "CHEST SINGLE VIEW:",
        "PORTABLE CHEST X-RAY:",
        "CHEST, SINGLE VIEW:",
        "CHEST CT TWO VIEWS:",
        "PORTABLE INDICATION:",
        "PORTABLE AP CHEST:",
        "INTRAOPERATIVE CHEST X-RAYS:",
    ]
    for _i in range(len(key_pool)):
        _key = key_pool[_i]
        if _key in report:
            report = report.replace(_key, "INDICATION:")

    key_pool_1 = [
        "RIGHT RIBS, FOUR VIEWS:",
        "RIBS",
        "PROVISIONAL FINDINGS IMPRESSION (PFI):",
        "CHEST",
        "BONE WINDOWS:",
        "IMPRESSION:  Successfully resolved CHF.  No new abnormalities.",
    ]
    for _i in range(len(key_pool_1)):
        _key = key_pool_1[_i]
        if _key in report:
            report = report.replace(_key, _key.lower())

    key_pool_2 = [
        "IMPRESSION:  PA and lateral chest radiographs.",
        "IMPRESSION:  PA and lateral radiographs of the chest.",
        "IMPRESSION:  Chest x-ray dated ___ at 18:10 p.m.",
        "IMPRESSION:  Portable frontal chest radiograph.",
        "INDICATION: Right lower chest in right upper quadrant pain.",
        "______________________________________________________________________________",
        "WET READ: ___ ___ ___ 9:14 AM",
        "MAIN REPORT  chest RADIOGRAPHS",
        "In comparison with the study of ___,",
        "In comparison with the study ___ ___,",
        "FINDINGS:  Comparison is made to the prior radiograph from ___ at",
        "As compared to ___ radiograph,",
    ]
    for _i in range(len(key_pool_2)):
        _key = key_pool_2[_i]
        if _key in report:
            report = report.replace(_key, "")

    key_pool_3 = [
        "An ET tube is present, tip approximately 3.7 cm above the carina.",
        "The lungs are well expanded and show a ___-mm left lower lobe opacity, which is",
        "Compared with ___, there is a new small right pneumothorax at the right",
        "The ET tube has been adjusted and currently the tip is approximately 3.7 cm",
        "Compared with ___ at 23:54 p.m., there is negligible interval change in",
        "Compared with ___, a right-sided Port-A-Cath has been placed, tip",
        "Compared with ___ and allowing for considerable technical differences, no",
        "A zone of minimally increased density at the right lung base is completely",
        "A PICC line is present.  The tip is not optimally visualized, but likely",
        "Impression lungs are well expanded and clear.  Cardiomediastinal and hilar",
        "Cardiomediastinal contours are normal.  The lungs are clear.  There is no",
        "Moderate to large left and small right pleural effusions are present.  The",
        "FRONTAL AND LATERAL VIEWS OF THE chest:  Opacity in the left lower lung is",
        "UPRIGHT PORTABLE chest RADIOGRAPH:  Moderate cardiomegaly is again seen.",
        "FRONTAL chest RADIOGRAPH:  The heart is top normal in size.  The hilar and",
        "There is no longer net volume loss in the left hemithorax, but there is severe",
        "Stable severe consolidation left lower lobe could be collapse or pneumonia.",
        "Compared with ___ at 00:35 a.m., the left-sided chest tube has been",
        "Lordotic positioning.  The patient is status post sternotomy.  An",
        "RECOMMENDATION(S):  As compared to ___, no relevant change is",
        "In comparison with the study of ___, there is little change and no",
        "FRONTAL chest RADIOGRAPH:  An endotracheal tube terminating 4.0 cm above the",
        "FRONTAL AND LATERAL VIEWS OF THE chest:  A post-pyloric enteric tube ends in",
        "COMPARISON:  Right PIC line ends in the region of the superior cavoatrial",
        "A multiloculated right pleural effusion appears minimally changed since",
        "Again seen is an ET tube, tip approximately 5.3 cm above the carina, and an NG",
        "Heart size and mediastinum are top normal.  Bilateral pleural effusions are",
        "The lungs are hyperinflated.  There is cardiomegaly, with an unfolded aorta.",
        "FRONTAL AND LATERAL chest RADIOGRAPHS:  Large loculated left pleural effusions",
        "Patient has had right mastectomy and axillary lymph node dissection.  AV left",
        "Heart size is normal.  Mediastinum is normal.  There is new pleural effusion,",
        "An ET tube is present, tip approximately 6.9 cm above the carina at the level",
        "The heart size is normal.  A right-sided aortic arch is again seen.  The hilar",
        "ET tube tip is approximately 3.6 cm above the carina.  NG tube passes below",
        "The lungs are low in volume and show a mild opacity in the right lower lobe",
        "Rotated positioning.  An ET tube is present.  The tip lies at the upper edge",
        "FRONTAL AND LATERAL VIEWS OF THE chest:  Compared to the prior study of ___, diffuse bilateral interstitial parenchymal abnormal",
        "RECOMMENDATION(S):  On ___, right middle and lower lobes were collapsed.",
        "PA AND LATERAL chest RADIOGRAPH:  Opacities involving the right mid lung zone",
        "FRONTAL chest RADIOGRAPH:  Enlarged left upper lobe opacities correspond to a",
        "RECOMMENDATION(S):  Multifocal infiltrative pulmonary abnormality has grown",
        "Again seen is dense opacity in the right mid zone with mixed lucency and",
        "Heart size is at the upper limits of normal.  No CHF, focal infiltrate or",
        "Tube tip is 5 cm above the carina.  NG tube is in the stomach.  The line is",
        "There is new left chest pigtail catheter.  Left pleural effusion is",
        "New asymmetrical elevation of the right hemidiaphragm is most consistent with",
        "Again seen is sternotomy, with AVR, mild cardiomegaly, left lower lobe",
        "INDICATION:  ___-year-old male patient with pace maker implant. AP single view",
        "Left internal jugular line has been inserted with its tip terminating in the",
        "The lungs are hyperinflated and the diaphragms are flattened, consistent with",
        "PA AND LATERAL chest RADIOGRAPH:  The background density of the lung",
        "FRONTAL AND LATERAL chest RADIOGRAPHS:  The heart is moderately enlarged.  The",
        "COMPARISON:  Heart size and mediastinum are stable.  Lungs are essentially",
        "COMPARISON:  As compared to the previous radiograph, the known and",
        "SEMIERECT AP VIEW OF THE chest:  The mild-to-moderate pulmonary edema and",
        "there is no relevant change.  Extensive bilateral areas of atelectasis,",
        "RECOMMENDATION(S):  Comparison to ___, 07:46.  A Swan-Ganz catheter was",
        "INDICATION:  Right chest tube remains in place, with a persistent",
        "UPRIGHT FRONTAL VIEW OF THE chest:  Moderate-to-severe cardiomegaly has",
        "Probable mild hyperinflation.  Heart size is borderline, with mild prominence",
        "NOTIFICATION:  Support and monitoring devices remain in standard position, and",
        "there is substantial enlargement of the cardiac silhouette with evidence of",
        "RECOMMENDATION(S):  Small right apical pneumothorax has decreased.  No",
        "FRONTAL AND LATERAL VIEWS OF THE chest:  Linear opacity at the left lung base",
        "FRONTAL AND LATERAL VIEWS OF THE chest:  The previously seen pulmonary edema",
        "AP AND LATERAL VIEWS OF THE chest:  There is no pneumothorax or pleural",
        "RECOMMENDATION(S):  Moderate bilateral pleural effusions, stable on the right,",
        "RECOMMENDATION(S):  Moderate pulmonary edema has improved.  Moderate pleural",
        "An ET tube is present, tip approximately 5.2 cm above the carina.  A right IJ",
        "A right PICC terminates at the mid SVC.  Mild cardiomegaly has slightly",
        "The heart is not enlarged.  There is no CHF, frank consolidation or effusion.",
        "NOTIFICATION:   And the CT study of that date, there is again a large",
        "No previous images.  The heart is normal in size and there is no evidence of",
        "RECOMMENDATION(S):  Dobhoff tube tip is in the stomach.  NG tube tip is in the",
        "The heart is not enlarged.  There is no CHF, focal consolidation, or gross",
        "PORTABLE UPRIGHT FRONTAL VIEW OF THE chest:  Interval development of prominent",
        "Compared with ___, I doubt significant interval change.  There is minimal",
        "RECOMMENDATION(S):  The patient is after esophagectomy.  Right chest tube is",
        "there is continued enlargement\n",
        "An ET tube is present, tip in satisfactory position 5.4 cm above the carina.",
        "COMPARISON:  Chest radiographs ___ through ___.  There has been no",
        "RECOMMENDATION(S):  There is no pneumothorax or pleural effusion.  Enlargement",
        "RECOMMENDATION(S):   there is no\n",
        "PORTABLE SUPINE FRONTAL VIEW OF THE chest:  There are median sternotomy wires.",
        "Multiple abandoned leads overlie the right chest.  A left-sided PICC line is",
        "Varices to ___.  Better seen on the lateral than on the frontal",
        "PA and lateral chest views were obtained with the patient in upright position.",
        "Cardiomediastinal contours are normal.  The lungs are hyperinflated.  Blunting",
        "UPRIGHT PORTABLE AP chest RADIOGRAPH:  Again seen is moderate stable",
        "Again seen are two left-sided chest tubes, and an NG tube extending beneath",
        "Dual-lead pacemaker again noted with lead tips over right atrium and right",
        "The right PICC line tip is at the level of superior SVC.  Cardiomegaly is",
        "PA AND LATERAL chest RADIOGRAPHS: A small-to-moderate right apical",
        "EXAMINATION: centered at the diaphragm excludes the right lateral chest and",
        "Compared with ___ at 4:11 a.m., I doubt significant interval change.  A\n",
        "Compared with ___, I doubt significant interval change. There is",
        "Compared with ___ at 14:39 p.m., I doubt significant interval",
        "Elevation of the right hemidiaphragm is again seen.  An air bronchogram is",
        "There is possible background hyperinflation, consistent with COPD.  There is",
        "COMPARISON:  Lungs are low in volume.  There is no focal pulmonary",
        "COMPARISON:  Chest radiographs since ___ most recently ___.",
        "TECHNIQUE:  Portable AP upright radiograph of the chest was obtained.  Lung",
        "FIDINGS: There are no significant interval changes in the lungs since ___.  Very mild pulmonary edema is similar.  Previously ap",
        "SINGLE AP chest, PA AND LATERAL VIEWS:  As noted, the patient is status post",
        "RECOMMENDATION(S):  Mild interstitial edema is unchanged.  Moderate",
        "FRONTAL chest RADIOGRAPH:  A large right pleural effusion has enlarged since",
        "As compared to the previous examination, the known pneumonia, also documented",
        "The heart size is normal.  Left perihilar and right paratracheal stripe",
        "Bilateral pigtail pleural catheters are in place without large pleural",
        "Marked cardiomegaly is accompanied by pulmonary vascular engorgement and",
        "Lordotic positioning, with low inspiratory volumes.",
        "COMPARISON:  Chest radiographs since ___, most recently ___.  Previous left lower lobe pneumonia has substantially improved.",
        "TECHNIQUE:  Two views of the chest were obtained.  The lungs are low in",
        "RECOMMENDATION(S):  No change since ___ in mild cardiomegaly and",
        "There appears to be interval worsening of the opacity seen in the right lung",
        "PORTABLE AP UPRIGHT chest RADIOGRAPH:  Right upper extremity PICC ends at the",
        "Status post sternotomy.  Lines and tubes are unchanged compared with ___",
        "Lordotic positioning and low inspiratory volumes.  A left-sided pacemaker is",
        "There is patchy opacity at the left base, with a small left effusion, new",
        "NOTIFICATION:  Small left pleural effusion is decreasing.  No pneumothorax.",
        "RECOMMENDATION(S):  Moderate interstitial pulmonary edema is new.  There is",
        "Improved left lung volume.  Right lung base opacity is stable.  Otherwise No",
        "Cardiomediastinal contours are normal.  The lungs are hyperinflated.  The",
        "PA AND LATERAL chest RADIOGRAPHS:  The cardiomediastinal and hilar contours",
        "In comparison with study of ___, there is little change.  The catheter",
        "Cardiac size is top normal.  The lungs are clear aside from minimal",
        "A left-sided PICC line has been placed, tip over distal SVC, unchanged.",
        "Right-sided internal jugular venous line is now seen with tip in the lower",
        "PA AND LATERAL chest RADIOGRAPH:  The cardiomediastinal and hilar contours are",
        "Frontal and lateral chest radiographs were obtained.  There is no significant",
        "Cardiac size is top normal.  Aside from a residual peripheral opacity within",
        "Interval decrease in the left pleural effusion, now small. No pneumothorax.",
        "Left IJ central line is present, tip in proximalmost SVC.  A nasoenteric type",
        "TECHNIQUE:  Monitoring and supporting devices are in standard position.",
        "There is mild cardiomegaly.  The aorta is unfolded.  No CHF, focal infiltrate,",
        "NOTIFICATION:  Mild enlargement of the hilar structures, confirming the",
        "Again seen is the tracheostomy tube, with focal dilatation of the tracheal air",
        "A tracheostomy is present.  The cardiomediastinal silhouette is markedly",
        "TWO VIEWS OF THE chest:  16 mm right upper lung nodule appears slightly",
        "PA and lateral chest views were obtained with patient in upright position.",
        "chest PORTABLE:  Comparison is made to ___ as well as chest CT of ___.",
        "Comparison is made with the prior chest x-ray of ___ immediately",
        "An ET tube is present The carina is not well defined, but the tip is in",
        "There has been interval progression of interstitial pulmonary edema, moderate",
        "Moderate to severe enlargement of the cardiac silhouette is obscured by right",
        "Right PICC terminates at the cavoatrial junction.  Clear lungs.",
        "There is bibasilar atelectasis, which likely accounts for some increased",
        "Three sequential chest radiographs show the Dobbhoff advanced through the",
        "There is patchy opacity in the left lower lobe seen on both frontal and",
        "chest, PA AND LATERAL:  The lungs are hyperexpanded, with apical hyperlucency",
        "An traceostomy tube is present, approximately 18 mm above the carina.",
        "In comparison with study of ___, there is some hyperexpansion of the lungs",
        "The heart is mildly enlarged.  The hilar and mediastinal contours are within",
        "On the lateral radiograph the location of the PICC line is in the right atrium",
        "There are low inspiratory volumes.  Heart size is borderline.  The mediastinal",
        "COMPARISON:  There no prior chest radiographs available for review.  Marked",
        "An NG tube is present, tip extending beneath diaphragm, off film.  A left IJ",
        "PORTABLE UPRIGHT AP VIEW OF THE chest:  Compared to the prior study performed",
        "PA AND LATERAL chest RADIOGRAPH:  Bilateral pleural effusions, large on the",
        "The possibility of an early infiltrate at the right base cannot be excluded.",
        "NOTIFICATION:  Increasing size of the cardiac silhouette.  The current image",
        "Cardiomegaly is noted, substantial, unchanged.  Pleural effusion is noted,",
        "The heart size is normal.  The hilar and mediastinal contours are within",
        "FRONTAL AND LATERAL chest RADIOGRAPHS:  A left-sided PICC terminates at the",
        "Previous vascular congestion and mild pulmonary edema have improved since",
        "Heart size is normal.  Mediastinum is normal.  Lungs are essentially clear.",
        "CLINICAL HISTORY:  There has been re-positioning of the nasogastric tube.  The",
        "COMPARISON:  ET tube tip is 5.4 cm above the carinal.  Lung volumes are low.",
        "HISTORY:  ___-year-old woman with colitis, rule out tuberculosis.  Low lung",
        "The lungs are hyperinflated, suggesting COPD.  The heart is not enlarged.",
        "There is background COPD.  The patient is status post sternotomy, with",
        "PORTABLE FRONTAL chest RADIOGRAPH:  Diffuse bilateral alveolar opacities",
        "The lungs are hyperinflated, consistent with COPD.  There is marked",
        "RECOMMENDATION(S):  Left pleural drainage catheter still in place.  No",
        "INDICATION:  There is coiling of the Dobbhoff catheter on one image and",
        "FRONTAL AND LATERAL chest RADIOGRAPHS:  Lungs are clear without confluent",
        "There are small bilateral effusions, slightly larger on the right, overall",
        "Compared with ___ at 13:51 p.m., the left pleural effusion has decreased in",
    ]
    for _i in range(len(key_pool_3)):
        _key = key_pool_3[_i]
        if _key in report:
            report = report.replace(_key, "FINDINGS: " + _key)

    key_pool_4 = [
        "COMPARISON:  Left pectoral infusion port catheter is the heart sharply bent at",
        "NOTE:  New volume loss, right upper lobe.  Mild cardiomegaly is stable.",
        "Impression transesophageal drainage tube ends in the upper portion of the",
        "COMPARISON:  The right chest tubes are in unchanged position. Right moderate",
    ]
    for _i in range(len(key_pool_4)):
        _key = key_pool_4[_i]
        if _key in report:
            report = report.replace(_key, "IMPRESSION: " + _key)

    report = report.replace("FINDINGS/IMPRESSION:", "FINDINGS:")
    report = report.replace(
        "Findings were discussed by Dr. ___ with Dr. ___ of pulmonology",
        "HISTORY: Findings were discussed by Dr. ___ with Dr. ___ of pulmonology",
    )
    report = report.replace(
        "Findings were discussed with the surgical attending",
        "HISTORY: Findings were discussed with the surgical attending",
    )
    report = report.replace(
        "IMPRESSION: Findings were discussed",
        "HISTORY: Findings were discussed",
    )
    report = report.replace(
        "IMPRESSION:  No cardiac enlargement, pulmonary congestion, or acute pulmonary",
        "No cardiac enlargement, pulmonary congestion, or acute pulmonary",
    )
    report = report.replace(
        "IMPRESSION:  No acute infiltrates or pulmonary congestion in this patient with",
        "No acute infiltrates or pulmonary congestion in this patient with",
    )
    report = report.replace(
        "acute\n symptoms.\n \n IMPRESSION: \n \n 1",
        "acute\n symptoms.\n \n 1",
    )
    report = report.replace(
        "IMPRESSION:  Bilateral airspace opacities and indistinctness of the hila most",
        "Bilateral airspace opacities and indistinctness of the hila most",
    )
    report = report.replace(
        "IMPRESSION:  Increasing pulmonary congestion with development of bilateral",
        "Increasing pulmonary congestion with development of bilateral",
    )
    report = report.replace(
        "FRONTAL SUPINE INDICATION:  A right subclavian catheter ends in the mid",
        "FINDINGS: A right subclavian catheter ends in the mid",
    )
    report = report.replace(
        "FRONTAL SUPINE INDICATION:  Endotracheal tube is low, ending 1.5 cm above",
        "FINDINGS: Endotracheal tube is low, ending 1.5 cm above",
    )
    report = report.replace(
        "PORTABLE FRONTAL chest RADIOGRAPH:  A right-sided chest tube has been placed",
        "FINDINGS: A right-sided chest tube has been placed",
    )
    report = report.replace(
        "PORTABLE FRONTAL chest RADIOGRAPH:  Low inspiratory lung volumes are unchanged",
        "FINDINGS: Low inspiratory lung volumes are unchanged",
    )
    report = report.replace(
        "Compared with ___ at 3:53 a.m., there is a new small right effusion and",
        "FINDINGS: There is a new small right effusion and",
    )
    report = report.replace(
        "Compared with ___ at 16:37 p.m., left apical pneumothorax is again seen,",
        "FINDINGS: Left apical pneumothorax is again seen,",
    )
    report = report.replace(
        "RECOMMENDATION(S):  Lung volumes are lower and mild interstitial pulmonary",
        "FINDINGS: Lung volumes are lower and mild interstitial pulmonary",
    )
    report = report.replace(
        "RECOMMENDATION(S):  Heart size is normal.  Lungs are clear.  There is no",
        "Heart size is normal.  Lungs are clear.  There is no",
    )
    report = report.replace(
        "Clear lungs. No pneumonia. No change from yesterday.",
        "FINDINGS: Clear lungs. No pneumonia.",
    )
    report = report.replace(
        "NOTIFICATION:  SEVERE CARDIOMEGALY IS CHRONIC. HETEROGENEOUS OPACIFICATION IN",
        "FINDINGS: SEVERE CARDIOMEGALY IS CHRONIC. HETEROGENEOUS OPACIFICATION IN",
    )
    report = report.replace(
        "FRONTAL INDICATION:  Endotracheal tube ends 3.8 cm above the carina.",
        "FINDINGS: Endotracheal tube ends 3.8 cm above the carina.",
    )
    report = report.replace(
        "FRONTAL chest RADIOGRAPH:  A right IJ catheter, transesophageal catheter,",
        "FINDINGS: A right IJ catheter, transesophageal catheter,",
    )
    report = report.replace(
        "chest PORTABLE:  Comparison is made to ___.  The lung volumes are low.",
        "FINDINGS: The lung volumes are low.",
    )
    report = report.replace(
        "FINDINGS:  FINDINGS: In comparison with the study of ___, there is little change and no",
        "FINDINGS: In comparison with the study of ___, there is little change and no",
    )
    report = report.replace(
        "PA and lateral views of the chest are obtained.  There is mild elevation of",
        "FINDINGS: There is mild elevation of",
    )
    report = report.replace(
        "I doubt significant interval change.  No definite pneumothorax is identified.",
        "FINDINGS: No definite pneumothorax is identified.",
    )
    report = report.replace(
        "TECHNIQUE:  Upright PA and lateral radiographs of the chest was obtained.  The",
        "FINDINGS: The",
    )
    report = report.replace(
        "COMPARISON:  The ET tube is been removed.  The NG tube is been removed.  The",
        "FINDINGS: The",
    )
    report = report.replace(
        "RECOMMENDATION(S):  A band of opacification base of the left lung is is",
        "FINDINGS: A band of opacification base of the left lung is",
    )
    report = report.replace(
        "improved.  Severe cardiomegaly is chronic.  ET tube in standard placement.",
        "FINDINGS: Severe cardiomegaly is chronic.  ET tube in standard placement.",
    )
    report = report.replace(
        "little interval change.  Again seen is probable background COPD, and",
        "FINDINGS: Again seen is probable background COPD, and",
    )
    report = report.replace(
        "Compared with ___ at 16:11 p.m., vertical atelectasis at the left base is",
        "FINDINGS: Vertical atelectasis at the left base is",
    )
    report = report.replace(
        "FINDINGS:  In comparison with the study of ___, there is some decrease in the",
        "FINDINGS: There is some decrease in the",
    )
    report = report.replace(
        "FINDINGS:  In comparison with the study of ___, there is again enlargement",
        "FINDINGS: There is again enlargement",
    )
    report = report.replace(
        "COMPARISON:  Since ___, left upper lobe opacity has resolved. Given the",
        "FINDINGS: Left upper lobe opacity has resolved. Given the",
    )
    report = report.replace(
        "In comparison with the study of ___, there is again some",
        "There is again some",
    )   
    report = report.replace(
        "FINDINGS:  In comparison with study of ___, there is little change and no",
        "FINDINGS: there is little change and no",
    )
    report = report.replace(
        "FINDINGS:  Comparison studies of ___, there is little change and no evidence",
        "FINDINGS: there is little change and no evidence",
    )
    report = report.replace(
        "In comparison to ___ radiograph, there has been development of",
        "FINDINGS: there has been development of",
    )   
    report = report.replace(
        "As compared to ___ radiograph, left pigtail pleural catheter",
        "FINDINGS: left pigtail pleural catheter",
    )
    report = report.replace(
        "INSPIRATION NO EVIDENCE OF VASCULAR CONGESTION, PLEURAL EFFUSION, OR ACUTE",
        "\nFINDINGS: NO EVIDENCE OF VASCULAR CONGESTION, PLEURAL EFFUSION, OR ACUTE",
    )
    report = report.replace(
        "FINDINGS:  In comparison with study of ___,",
        "FINDINGS:",
    )
    report = report.replace(
        "In comparison to ___ chest radiograph, cardiomegaly is accompanied by",
        "FINDINGS: cardiomegaly is accompanied by",
    )
    report = report.replace(
        "As compared to ___, cardiomegaly and pulmonary vascular congestion",
        "FINDINGS: cardiomegaly and pulmonary vascular congestion",
    )
    report = report.replace(
        "FINSINGS:  The opacities within the right lung have improved.  There is also",
        "FINDINGS: The opacities within the right lung have improved.  There is also",
    )
    report = report.replace(
        "As compared to ___ radiograph, pulmonary edema is more severe,",
        "FINDINGS: pulmonary edema is more severe,",
    )
    report = report.replace(
        "Compared with ___ at ___ p.m., the chest tube has been removed.  A small",
        "FINDINGS: A small",
    )
    report = report.replace(
        "COMPARISON:  Prior chest radiographs are not available.  Lungs are fully",
        "FINDINGS: Lungs are fully",
    )
    report = report.replace(
        "As compared ___ chest radiograph, cardiomegaly is accompanied by",
        "FINDINGS: cardiomegaly is accompanied by",
    )
    report = report.replace(
        "FINDINGS:   FINDINGS: there is continued enlargement",
        "FINDINGS: there is continued enlargement",
    )
    report = report.replace(
        "FINDINGS: FINDINGS: there is continued enlargement",
        "FINDINGS: there is continued enlargement",
    )
    report = report.replace(
        "In comparison study of ___, the cardiac silhouette remains within",
        "FINDINGS: the cardiac silhouette remains within",
    )
    report = report.replace(
        "In comparison with study ___ ___, there is again hyperexpansion of the",
        "FINDINGS: there is again hyperexpansion of the",
    )
    report = report.replace(
        "In comparison to ___ radiograph, mild cardiomegaly is accompanied by",
        "FINDINGS: mild cardiomegaly is accompanied by",
    )
    report = report.replace(
        "is mild, improved since ___ and there is no edema, pleural effusion, or",
        "is mild, and there is no edema, pleural effusion, or",
    )
    report = report.replace(
        "Is some with the study of ___, there is again substantial enlargement of",
        "FINDINGS: there is again substantial enlargement of",
    )
    report = report.replace(
        "In comparison with study of ___, there is little change and no",
        "FINDINGS: no",
    )
    report = report.replace(
        "is little change and no evidence of acute focal pneumonia, vascular",
        "\nFINDINGS: no evidence of acute focal pneumonia, vascular",
    )
    report = report.replace(
        "In comparison with study of ___, there is no evidence of acute pneumonia,",
        "\nFINDINGS: there is no evidence of acute pneumonia,",
    )
    report = report.replace(
        "In comparison to ___ chest radiograph, a pigtail pleural catheter",
        "FINDINGS: a pigtail pleural catheter",
    )
    report = report.replace(
        "In comparison with the study of to ___, there again are bilateral",
        "\nFINDINGS: there again are bilateral",
    )
    report = report.replace(
        "As compared to ___, cardiac silhouette remains markedly enlarged",
        "\nFINDINGS: cardiac silhouette remains markedly enlarged",
    )
    report = report.replace(
        "In comparison with the study obtained ___ min previously, there is an placement",
        "\nthere is an placement",
    )
    report = report.replace(
        "Again seen is the small right apical pneumothorax, overall similar, but",
        "\nFINDINGS: Again seen is the small right apical pneumothorax, overall similar.",
    )
    report = report.replace(
        "In comparison to ___ chest radiograph, pulmonary vascular",
        "\nFINDINGS: pulmonary vascular",
    )
    report = report.replace(
        "Since the 11:04 a.m. examination, there has been marked interval increase in",
        "\n FINDINGS: there has been marked interval increase in",
    )
    report = report.replace(
        "In comparison to ___ chest radiograph, cardiomediastinal contours",
        "FINDINGS: cardiomediastinal contours",
    )
    report = report.replace(
        "Aside from mild linear atelectasis at the lung bases lungs are clear.  There",
        "\nFINDINGS: Aside from mild linear atelectasis at the lung bases lungs are clear.  There",
    )
    report = report.replace(
        "Moderate left pleural effusion and left basal atelectasis are unchanged.",
        "\nFINDINGS: Moderate left pleural effusion and left basal atelectasis are unchanged.",
    )
    report = report.replace(
        "FINDINGS:  FRONTAL INDICATION:  Dobbhoff tube ends in the stomach.  Left",
        "\nFINDINGS: Dobbhoff tube ends in the stomach.  Left",
    )
    report = report.replace(
        "In comparison to ___ chest radiograph, cardiomegaly and pulmonary",
        "cardiomegaly and pulmonary",
    )
    report = report.replace(
        "In comparison to ___ chest radiograph, a nonspecific left",
        "\nFINDINGS: a nonspecific left",
    )
    report = report.replace(
        "COMPARISON: Portable chest radiograph ___ at 14:19.\n \n PORTABLE AP chest RADIOGRAPH:",
        "\n..\nFINDINGS: ",
    )
    report = report.replace(
        "In comparison with study of ___, the cardiac silhouette remains",
        "\nFINDINGS: the cardiac silhouette remains",
    )
    report = report.replace(
        "presumably atelectasis.  Mild to moderate left lower lobe atelectasis has",
        "\n FINDINGS: Mild to moderate left lower lobe atelectasis has",
    )
    report = report.replace(
        "Previous pulmonary vascular congestion has improved slightly.  There is no",
        ".\n FINDINGS: There is no",
    )
    report = report.replace(
        "Compared to the prior radiograph of ___, there is been little",
        "\nFINDINGS: there is been little",
    )
    report = report.replace(
        "In comparison to ___ radiograph, the lungs remain clear, with no",
        "\nFINDINGS: the lungs remain clear, with no",
    )
    report = report.replace(
        "PROGRESSED SINCE ___, BUT THERE IS NO PULMONARY EDEMA PLEURAL EFFUSION OR",
        "\nFINDINGS: THERE IS NO PULMONARY EDEMA PLEURAL EFFUSION OR",
    )
    report = report.replace(
        "change is detected.  Again seen are sternotomy wires.  The cardiomediastinal",
        "\n.\n FINDINGS: Again seen are sternotomy wires.  The cardiomediastinal",
    )
    report = report.replace(
        "detected.  Otherwise, I doubt significant interval change.",
        "\nFINDINGS: No pneumothorax is detected.",
    )
    report = report.replace(
        "THE LUNGS CONSISTENT WITH CHRONIC PULMONARY DISEASE, BUT NO EVIDENCE OF ACUTE",
        "\nFINDINGS: NO EVIDENCE OF ACUTE",
    )
    report = report.replace(
        "As compared to the recent chest radiograph of ___, there remains a",
        "\nFINDINGS: there remains a",
    )
    report = report.replace(
        "IMPRESSION:  No acute cardiopulmonary process\n",
        "IMPRESSION:  No acute cardiopulmonary process.\n",
    )
    report = report.replace(
        "In comparison a ___ chest radiograph, the lungs remain clear, with",
        "the lungs remain clear, with",
    )
    report = report.replace(
        "In comparison to ___ radiograph, cardiomegaly is accompanied by",
        "cardiomegaly is accompanied by",
    )

    # if "IMPRESSION: Chest pain." in report:
    #     return report.replace("IMPRESSION: Chest pain.", "INDICATION: _.")

    # if "FINDINGS:  PA and two lateral radiographs of the chest were obtained and" in report:
    #     return report.replace("FINDINGS:  PA and two lateral radiographs of the chest were obtained", "")

    # if "INDICATION:\n \n There is a right chest port," in report:
    #     return report.replace("INDICATION:\n \n There is a right chest port,", "FINDINGS: There is a right chest port,")

    report = replace_appearance(report, "INDICATION:", "FINDINGS:", 2)
    report = replace_appearance(report, "FINDINGS:", "HISTORY:", 1)
    if "FINDINGS:" not in report:
        report = replace_appearance(report, "IMPRESSION:", "FINDINGS:", 1)

    return report


def get_mimic_data(text_filepath):
    # print(f"text_filepath: {text_filepath}")

    path_to_report = text_filepath
    if not os.path.exists(path_to_report):
        print(f"path_to_report: {path_to_report}")
        input()
        return None

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    report = customize_report(report, text_filepath)
    # print([report,])

    sections, section_names, _ = sp.section_text(report)

    # if ("findings" in section_names) or ("impression" in section_names):
    #     return None

    # print("###### before ######")
    # print(report)

    # print(f"section_names: {section_names}")
    # for _key in section_names:
    #     _index = section_names.index(_key)
    #     _part = sections[_index]
    #     print(f">>>>>> {_key}: {_part}")

    duplicated_names = find_duplicates(section_names)
    # if len(duplicated_names) > 0:
    #     print(section_names)
    #     input()
    #     return

    if len(duplicated_names) > 0 and (
        "findings" in duplicated_names or "impression" in duplicated_names
    ):

        print("###### before ######")
        # print([report,])
        print(report)

        print(f"section_names: {section_names}")
        # input()

    out_report = ""
    for _key in section_names:
        if _key in [
            "finding",
            "findings",
            "impression",
            "impressions",
            "last_paragraph",
            "report",
        ]:
            _index = section_names.index(_key)
            _part = sections[_index]
            out_report += _part
            # print(f">>>>>> {_key}: {_part}")

    out_report = remove_newline(out_report)
    out_report = normalize_spaces(out_report)
    out_report = replace_abbreviations(out_report)

    sentences = split_into_sentences(out_report)
    sentences = remove_before_colon(sentences)
    sentences = remove_sentences_with_underscore(sentences)
    sentences = refine_numbered_sentences(sentences)
    sentences = skip_to_first_letter(sentences)
    sentences = remove_single_char_sentences(sentences)
    sentences = capitalize_first_letter(sentences)
    sentences = add_period_if_missing(sentences)
    sentences = remove_duplicate_sentences(sentences)
    sentences = remove_question_sentences(sentences)

    if len(duplicated_names) > 0 and (
        "findings" in duplicated_names or "impression" in duplicated_names
    ):
        print("###### after ######")
        print("\n".join(sentences))
        input()

    # if len(sentences) < 3:
    #     input()

    # input()

    if int(sys.argv[2]) > 0:
        print(report)

    if len(sentences) > 0:
        return " ".join(sentences)
    else:
        print(text_filepath)
        print(report)
        input()

    return None


# list_filepath = (
#     "/localhome/local-dongy/projects/vlm/lists/mimic-cxr-2.0.0-test_v0_pairs160.txt"
# )
# output_dir = "/localhome/local-dongy/projects/vlm/text_gt/test"
# list_filepath = (
#     "/localhome/local-dongy/projects/vlm/lists/mimic-cxr-2.0.0-train_v0_pairs1024.txt"
# )

# list_filepath = (
#     "/localhome/local-dongy/projects/vlm/lists/mimic-cxr-2.0.0-test_v0_pairs.txt"
# )
# output_dir = "/localhome/local-dongy/projects/vlm/text_gt/test"
list_filepath = (
    "/localhome/local-dongy/projects/vlm/lists/mimic-cxr-2.0.0-train_v0_pairs.txt"
)
output_dir = "/localhome/local-dongy/projects/vlm/text_gt/training_1"

report_dir = "/mnt/drive1/reports"

with open(list_filepath, "r") as file:
    filepaths = file.readlines()
filepaths = [_item.strip() for _item in filepaths]
# filepaths = [_item for _item in filepaths if _item.endswith(".txt")]

# random.shuffle(filepaths)
# print(filepaths)

num_cases = len(filepaths) // 2

for _i in range(int(sys.argv[1]) - 1, num_cases):
    os.system("clear")

    print(f"{_i + 1} / {num_cases}\n")

    image_filename = filepaths[2 * _i]
    text_filepath = filepaths[2 * _i + 1]
    # text_filepath = filepaths[_i]

    new_text = get_mimic_data(os.path.join(report_dir, text_filepath))
    if new_text == None:
        continue

    if int(sys.argv[2]) > 0:
        print(new_text)
        input()

    output_filepath = os.path.join(output_dir, image_filename + ".txt")
    with open(output_filepath, "w") as output_file:
        output_file.write(new_text)
