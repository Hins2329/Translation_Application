package com.translation_service.translation_service.translate.controller;

import com.translation_service.translation_service.translate.application.TranslateApplicationService;
import com.translation_service.translation_service.translate.domain.model.TranslationRequest;
import com.translation_service.translation_service.translate.domain.model.TranslationResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/translate")
public class TranslateController {
    private final TranslateApplicationService translateApplicationService;

    public TranslateController(TranslateApplicationService translateApplicationService) {
        this.translateApplicationService = translateApplicationService;
    }

    @PostMapping
    public ResponseEntity<String> getTranslation(@RequestBody TranslationRequest request) {
        TranslationResponse response = translateApplicationService.translate(request.getSentence());
        System.out.println(response.toString());

        return ResponseEntity.ok("Sentence went: " + response.getTranslation());
    }
}
