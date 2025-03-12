package com.translation_service.translation_service.translate.infras.feign;

import com.translation_service.translation_service.translate.domain.model.TranslationRequest;
import com.translation_service.translation_service.translate.domain.model.TranslationResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;

@FeignClient(name = "transformer-service")
public interface TranslateServiceClient {


    @PostMapping("/TransformerTranslate")
    TranslationResponse translate(@RequestBody TranslationRequest request);


}
