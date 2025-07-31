ragas_data_set = {
    "user_input": [
        "Wie is de product owner van AGDP en kan je de contactgegevens geven?",
        "Wie is de product owner van AGPR en kan je de contactgegevens geven?",
        "Kan je me alle contactgegevens geven voor het fragment UBO-register, ook de SNOW groep.",
        "Een burger krijgt non stop nieuwe documenten, wat moet ik hiermee doen?",
        "Iemand heeft gevraagd naar de auditlogs wat moet ik hiermee doen?",
        "Hoe of waar moet ik mijn tijd loggen voor de support?",
        "Ik heb een vraag gekregen van een belastingplichtige wat moet ik hiermee doen?",
        "Een gebruiker heeft geen toegang tot een functionaliteit wat moet ik hiermee doen",
        "Er is een probleem met attest service, wie kan ik hiervoor contacteren?",
        "Hoe doe ik een repush voor RV?",
        "Iemand wil een document verwijderen dat ik hij heeft opgeladen via MyMinfin, wat kan ik hierop antwoorden?",
        "Een burger kan werkt bij een onderneming, maar kan niet inloggen in de naam van de onderneming, wat moet ik hierop antwoorden?",
        "Wat zijn de contactgegevens voor de hypoimage service?"
    ],
    "retrieved_contexts": [],
    "answer": [],
    "ground_truth": [
        "De product owner van AGDP is Kelly Fugarino en je kan haar contacteren via myminfin.agdp_aapd@temp.minfin.be",
        "De product owner van AGPR is Adil Soussi Nachit en je kan hem contacteren via aaii-agpr.myminfin@minfin.fed.be",
        "Fragment: UBO-register – Contactgegevens\n- ICT\n  - Group SNOW: TRES_GRP_CIFI\n  - Service manager: Christophe MARTIN (back-up: Vivian VACHAUDEZ)\n- Business\n  - Business analyst: Sébastien GUILLAUME",
        "Ce n’est pas à MMF de traiter ; il faut l’envoyer à Florent Defour (dev), Cédric Micha (ABA) et Patrick Bottu (SM, équipe MyRent).",
        "For audit logs in the ACC environment, please contact Michel Molitor. If he is unavailable, you can open a ticket with IAM ICT. For audit logs in the PROD environment, contact the IAM team of the responsible pillar.",
        """Where to log support time (from most preferred to least preferred):
    
        1. Related ticket
           - Project: MYMINFIN
           - Log time to an existing issue whenever possible.
    
        2. New bug
           - Project: MYMINFIN
           - If the reported problem requires creating a bug issue, log your time in an “analysis” subtask.
    
        3. Business Support
           - Fisc: [MYMINFIN-9958] [support] MYMINFIN 3rd line AGFISC/AAFISC - JIRA Cloud – ACTIVE
           - Rec: MYMINFIN-12102: Recouvrement Support – ACTIVE
           - Patdoc: MYMINFIN-12104: PatDoc Support – ACTIVE
           - Douane: MYMINFIN-12106: Douane Support – ACTIVE
           - Trésorerie: MYMINFIN-12108: Trésorerie Support – ACTIVE
           - (Until end of August, except FISC)
    
        4. Transversal Support (last resort if none of the above applies)
           - Transversal: MYMINFIN-12182: Transversal Support – ACTIVE
           - ICT: MYMINFIN-12110: Support Day – ACTIVE
           - (Until end of August)""",
        "Als Stafdienst ICT kunnen wij uw vraag jammer genoeg niet behandelen. Voor dergelijke vragen moeten we u doorverwijzennaar de contact center van de FOD Financiën (02/572.57.57 normaal tarief).",
        """Vragen met betrekking tot toegangsrechten moeten voortaan worden doorgestuurd naar de algemene mailboxen van de IAM-cellen.
Let op: deze adressen MOGEN NIET aan klanten worden meegedeeld.

Fiscaliteit: support-egov.agfisc@minfin.fed.be

Invordering & Innen: support-egov.agpr@minfin.fed.be

Patrimoniumdocumentatie: support-egov.agdp@minfin.fed.be

Douane & Accijnzen: support-egov.da@minfin.fed.be

Thesaurie: support-egov.thesaurie@minfin.fed.be

Algemeen: support-egov@minfin.fed.be""",
        "Je kan contact ponemen met het development team via grp-minfin-finattest-devteam@minfin.fed.be",
        """
        Hier zijn de stappen die gevolgd moeten worden wanneer het RV-team vraagt om een “repush” uit te voeren:
        
        1. Open MyMinfin Civil Servant in een incognitovenster.
        
        2.Open een tweede tabblad en zet er alvast: http://www.finbel.intra/
        
        3. Zorg ervoor dat u over de volgende informatie beschikt:
        
            - Application Type
            
            - Status
            
            - ID
        
        4. In het tweede tabblad, plak een link met de volgende structuur achter: http://www.finbel.intra/
        
        Repush via ID: myminfin-rest/myminfin-files-stirint/put-in-queue/{applicationType}/{status}/{id}
        
        Voorbeeld: myminfin-rest/myminfin-files-stirint/put-in-queue-by-id/WITHHOLDING_TAX/VALIDATED/204832
        
        Opmerking: Als u het volledige adres van de aanvrager krijgt, let er dan op dat er niet te veel spaties in staan.
        
        5. Druk op ENTER.
        
        6. Als alles goed verlopen is, geeft het systeem een succesbericht terug dat gecombineerd moet zijn met de melding 'queued files: 1'.
        
        Voorbeeld: {"restMessages":[{"type":"SUCCESS","bundleKey":"queued files : 1","sticky":false,"date":"2023-07-03T13:41:29.848+0000"}],"mustRedirectToHome":false,"sessionExpired":false,"error":false}
         """,
        """Je kan het volgende antwoorden: MYMINFIN slaat geen gegevens op. Als het proces niet toelaatom te annuleren, moet de gebruiker contact opnemen met deafdeling die het document heeft ontvangen.
    Ik vermoed dat dit document is verstuurd via "Antwoord op een brief" - in dit geval moet de klant de gebruikte driecijferige codemeedelen en moet je het ticket doorsturen naar het SOH-team.Die kunnen je vertellen met welke bevoegde service contactmoet worden opgenomen .
    Wij kunnen u niet helpen.""",
        """Indien u niet de wettelijke vertegenwoordiger van deonderneming bent, hebt u een rol nodig om in te loggen inMYMINFIN in naam van uw eigen onderneming (inclusieffilialen).
            U wilt in MYMINFIN toegang tot :
            de berichten van de onderneming, dan hebt u de rol FODFIN Berichten van de onderneming nodig;
            de documenten en formaliteiten van de onderneming, danhebt u meestal genoeg aan de rol FOD Fin Aanstelling EigenOnderneming met één of meerdere parameters in functievan uw verantwoordelijkheden (bv. Om de BTW formaliteitente beheren, hebt u de parameter ‘Aanstelling BTW' nodig).
            Indien uw onderneming meerdere filialen telt, hebt u enkel derol nodig die volstaat voor elk van die ondernemingen. Vergeethet begrip ‘mandaten'. Die zijn niet nodig als het om uw eigenonderneming (inclusief filialen) gaat.
            Voor meer informatie kan u de volgende websites raadplegen :
            welke rol ik moet toekennen : 
            Rollen
            hoe toekennen of aanvragen van een rol :
            Beheer van detoegangsaanvragen in de rollenadministratie- RMA - Loginhulp.be
            Eens dit in orde is, logt u aan op MyMinfin en identificeert uzich “in naam van een onderneming”. Selecteer uw eigen onderneming (of een filiaal).""",
        """service : AGDP - HYPOIMAGE
contacts : rzsj.operationeleondersteuning@minfin.fed.be ou PATDOC-RZSJ-Hypo (1e ligne)
 - andy.vanrompuy@minfin.fed.be (ICT - servicemanager ad interim)
 - dirk.vanherrewegen@minfin.fed.be (ICT -développeur)"""
    ]
}
