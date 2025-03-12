package com.translation_service.translation_service.translate.application;

import com.translation_service.translation_service.translate.domain.model.TranslationResponse;
import com.translation_service.translation_service.translate.domain.service.TranslateService;
import org.springframework.stereotype.Component;

@Component
public class TranslateApplicationService {


    private final TranslateService translateService;

    public TranslateApplicationService(TranslateService translateService) {
        this.translateService = translateService;
    }

    public TranslationResponse translate(String sentence){
        return translateService.translate(sentence);
    }


}
