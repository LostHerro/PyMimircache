//
//  glib_related.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "glib_related.h"

void simple_key_value_destroyed(gpointer data) {
    free(data);
}

void gqueue_destroyer(gpointer data) {
    g_queue_free_full(data, simple_key_value_destroyed);
}