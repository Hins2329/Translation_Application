package com.translation_service.translation_service.user.infras.Mapper;

import com.translation_service.translation_service.common.model.User;
import com.translation_service.translation_service.user.infras.persistence.UserDocument;
import org.springframework.stereotype.Component;
import org.bson.Document;

import java.util.List;


@Component
public class UserMapper {

    public static UserDocument fromMongoDocument(List<Document> docs) {
        if (docs == null || docs.isEmpty()) return null;

        // If there are multiple documents, filter out the one with op:'d' and keep op:'c'
        Document docToUse = docs.size() > 1
                ? docs.stream().filter(d -> !"d".equals(((Document) d.get("payload")).getString("op"))).findFirst().orElse(null)
                : docs.get(0);

        if (docToUse == null) return null;

        Document payload = (Document) docToUse.get("payload");
        Document after = (Document) payload.get("after");
        Document before = (Document) payload.get("before");

        Document sourceData = (after != null) ? after : before;
        if (sourceData == null) return null;

        UserDocument userDocument = new UserDocument();
//        userDocument.setId(sourceData.getInteger("id"));
        userDocument.setId(sourceData.getLong("id"));
        userDocument.setUsername(sourceData.getString("username"));
        userDocument.setPassword(sourceData.getString("password"));
        userDocument.setRole(sourceData.getString("role"));
        userDocument.setSentences(sourceData.getList("sentences", String.class));


        return userDocument;
    }

    public static User toUser(UserDocument userDoc) {
        if (userDoc == null) return null;
        return new User(userDoc.getId(), userDoc.getUsername(), userDoc.getRole(), userDoc.getSentences(), userDoc.getPassword());
    }

    public static UserDocument toUserDocument(User user) {
        if (user == null) return null;
        return new UserDocument(user.getId(), user.getUsername(), user.getPassword(), user.getRole(), user.getSentences());
    }




}
