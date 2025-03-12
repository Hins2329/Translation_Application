package com.translation_service.translation_service.translate.domain.service;

import com.translation_service.translation_service.translate.domain.model.TranslationRequest;
import com.translation_service.translation_service.translate.domain.model.TranslationResponse;
import com.translation_service.translation_service.translate.infras.feign.TranslateServiceClient;
import org.springframework.stereotype.Service;

@Service
public class TranslateService {
    private final TranslateServiceClient translateServiceClient;

    public TranslateService(TranslateServiceClient translateServiceClient) {
        this.translateServiceClient = translateServiceClient;
    }


    public TranslationResponse translate(String sentence){
        TranslationRequest translationRequest = new TranslationRequest(sentence);
        return translateServiceClient.translate(translationRequest);
    }


}
